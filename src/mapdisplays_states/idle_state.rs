use ffmpeg_next as ffmpeg;
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ndarray::Array3;
use numpy::PyArray3;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::DisplayState;

pub struct IdleState {
    current_frame: Arc<Mutex<Array3<u8>>>,
    frame_id: Arc<AtomicU16>,
    running: Arc<AtomicBool>,
    pub target_fps: f64,
}

impl IdleState {
    pub fn new(width: u16, height: u16, video_path: String, target_fps: f64) -> PyResult<Arc<Self>> {
        // Initialize FFmpeg libraries
        ffmpeg::init().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let state = Arc::new(IdleState {
            current_frame: Arc::new(Mutex::new(Array3::zeros((height as usize, width as usize, 3)))),
            frame_id: Arc::new(AtomicU16::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            target_fps,
        });

        let state_clone = state.clone();
        thread::spawn(move || {
            while state_clone.running.load(Ordering::Acquire) {
                if let Err(e) = state_clone.run_decode_loop(&video_path, width, height) {
                    eprintln!("[IdleState] Decode error: {:?}", e);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });

        Ok(state)
    }

    fn run_decode_loop(&self, path: &str, w: u16, h: u16) -> Result<(), ffmpeg::Error> {
        let mut ictx = ffmpeg::format::input(&path)?;
        let input = ictx.streams().best(Type::Video).ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_index = input.index();

        let context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context.decoder().video()?;

        let avg_fps_raw = input.avg_frame_rate();
        let native_fps = avg_fps_raw.0 as f64 / avg_fps_raw.1 as f64;
        
        let drop_factor = (native_fps / self.target_fps).max(1.0);
        let frame_duration = Duration::from_secs_f64(1.0 / self.target_fps);

        let mut scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            w as u32,
            h as u32,
            Flags::POINT,
        )?;

        let mut native_frame_count: u64 = 0;
        let mut next_render_time = Instant::now();

        for (stream, packet) in ictx.packets() {
            if !self.running.load(Ordering::Acquire) { break; }
            if stream.index() == video_index {
                decoder.send_packet(&packet)?;
                let mut decoded = Video::empty();

                while decoder.receive_frame(&mut decoded).is_ok() {
                    native_frame_count += 1;

                    if (native_frame_count as f64 % drop_factor) < 1.0 {
                        let mut rgb_frame = Video::empty();
                        scaler.run(&decoded, &mut rgb_frame)?;

                        let data = rgb_frame.data(0);
                        let stride = rgb_frame.stride(0);
                        let mut frame_array = Array3::zeros((h as usize, w as usize, 3));

                        for y in 0..h as usize {
                            let start = y * stride;
                            let row = &data[start..start + (w as usize * 3)];
                            for (x, chunk) in row.chunks_exact(3).enumerate() {
                                frame_array[[y, x, 0]] = chunk[0];
                                frame_array[[y, x, 1]] = chunk[1];
                                frame_array[[y, x, 2]] = chunk[2];
                            }
                        }

                        {
                            let mut lock = self.current_frame.lock();
                            *lock = frame_array;
                        }
                        self.frame_id.fetch_add(1, Ordering::Release);

                        // Sync to real-world clock
                        next_render_time += frame_duration;
                        let now = Instant::now();
                        if next_render_time > now {
                            thread::sleep(next_render_time - now);
                        } else if now - next_render_time > Duration::from_secs(1) {
                            // If we are more than 1s behind, reset the clock to avoid "speed-up" effect
                            next_render_time = now;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Drop for IdleState {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
    }
}

impl DisplayState for IdleState {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        let frame_data = self.current_frame.lock().clone();
        let id = self.frame_id.load(Ordering::Acquire);
        let py_array = PyArray3::from_array(py, &frame_data);
        Ok((py_array, id))
    }

    fn stop(&self) -> PyResult<()> {
        self.running.store(false, Ordering::Release);
        Ok(())
    }
}

#[pyclass(extends = crate::PyDisplayState)]
pub struct PyIdleState {
    state: Arc<IdleState>,
}

#[pymethods]
impl PyIdleState {
    #[new]
    fn new(width: u16, height: u16, video_path: String, target_fps: f64) -> PyResult<(Self, crate::PyDisplayState)> {
        let arc_state = IdleState::new(width, height, video_path, target_fps)?;
        
        let display_state = crate::PyDisplayState {
            inner: Arc::new(Mutex::new(
                Box::new(ArcIdleStateWrapper(arc_state.clone())) as Box<dyn DisplayState>
            )),
        };
        
        Ok((PyIdleState { state: arc_state }, display_state))
    }

    #[getter]
    fn target_fps(&self) -> f64 {
        self.state.target_fps
    }

    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        self.state.get_full_frame(py)
    }

    fn stop(&self) -> PyResult<()> {
        self.state.stop()
    }
}

struct ArcIdleStateWrapper(Arc<IdleState>);

impl DisplayState for ArcIdleStateWrapper {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        self.0.get_full_frame(py)
    }

    fn stop(&self) -> PyResult<()> {
        self.0.stop()
    }
}