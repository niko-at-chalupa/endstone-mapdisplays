use ffmpeg_next as ffmpeg;
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ndarray::Array3;
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::thread;
use numpy::PyArray3;
use crate::DisplayState;
use pyo3::Bound;

pub struct IdleState {
    current_frame: Arc<Mutex<Array3<u8>>>,
    frame_id: Arc<AtomicU16>,
    running: Arc<AtomicBool>,
}

impl IdleState {
    pub fn new(width: u16, height: u16, video_path: String) -> PyResult<Arc<Self>> {
        ffmpeg::init().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let state = Arc::new(IdleState {
            current_frame: Arc::new(Mutex::new(Array3::zeros((height as usize, width as usize, 3)))),
            frame_id: Arc::new(AtomicU16::new(0)),
            running: Arc::new(AtomicBool::new(true)),
        });

        let state_clone = state.clone();
        thread::spawn(move || {
            while state_clone.running.load(Ordering::Acquire) {
                if let Err(e) = state_clone.decode_video(&video_path, width, height) {
                    eprintln!("[IdleState] FFmpeg Error: {}", e);
                    thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        });

        Ok(state)
    }

    fn decode_video(&self, path: &str, w: u16, h: u16) -> Result<(), ffmpeg::Error> {
        let mut ictx = ffmpeg::format::input(&path)?;
        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        
        let video_stream_index = input.index();
        let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;

        // Scaler to convert whatever the video is into RGB24 at your target dimensions
        let mut scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            w as u32,
            h as u32,
            Flags::BILINEAR,
        )?;

        let mut frame_id_local = 0;

        for (stream, packet) in ictx.packets() {
            if !self.running.load(Ordering::Acquire) { break; }
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;
                let mut decoded = Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut rgb_frame = Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;

                    // Move data into ndarray
                    let data = rgb_frame.data(0);
                    let stride = rgb_frame.stride(0);
                    let mut frame_array = Array3::zeros((h as usize, w as usize, 3));
                    
                    // Copy line by line to account for stride/padding
                    for y in 0..h as usize {
                        let start = y * stride;
                        let end = start + (w as usize * 3);
                        let row = &data[start..end];
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
                    
                    // Logic for frame timing (FPS control) would go here
                    // Simplified: sleep for ~33ms
                    thread::sleep(std::time::Duration::from_millis(30));
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

#[pyclass(extends = crate::PyDisplayState)]
pub struct PyIdleState {
    state: Arc<IdleState>,
}

#[pymethods]
impl PyIdleState {
    #[new]
    fn new(width: u16, height: u16, video_path: String) -> PyResult<(Self, crate::PyDisplayState)> {
        let arc_state = IdleState::new(width, height, video_path)?;
        
        // We wrap the Arc in our trait-object wrapper so it can be used 
        // by the generic PyDisplayState logic
        let display_state = crate::PyDisplayState {
            inner: Arc::new(Mutex::new(
                Box::new(ArcIdleStateWrapper(arc_state.clone())) as Box<dyn DisplayState>
            )),
        };
        
        Ok((PyIdleState { state: arc_state }, display_state))
    }

    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        self.state.get_full_frame(py)
    }

    fn stop(&self) -> PyResult<()> {
        self.state.stop()
    }
}

// These traits ensure the PyO3 bridge can access the methods defined on IdleState
struct ArcIdleStateWrapper(Arc<IdleState>);

impl DisplayState for ArcIdleStateWrapper {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        self.0.get_full_frame(py)
    }

    fn stop(&self) -> PyResult<()> {
        self.0.stop()
    }
}

impl DisplayState for IdleState {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        let frame_data = self.current_frame.lock().clone();
        let id = self.frame_id.load(Ordering::SeqCst);
        let py_array = PyArray3::<u8>::from_array(py, &frame_data);
        Ok((py_array, id))
    }

    fn stop(&self) -> PyResult<()> {
        self.running.store(false, Ordering::Relaxed);
        Ok(())
    }
}