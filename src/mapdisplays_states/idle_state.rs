use numpy::PyArray3;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use std::path::PathBuf;
use ndarray::Array3;
use parking_lot::Mutex;
use crate::DisplayState;

pub struct IdleState {
    current_frame: Arc<Mutex<Array3<u8>>>,
    frame_id: Arc<AtomicU16>,
    running: Arc<AtomicBool>,
    temp_file: Option<PathBuf>,
}

impl IdleState {
    pub fn new(width: u16, height: u16, video_path: String) -> PyResult<Arc<Self>> {
        let (actual_path, temp_file) = Self::prepare_resource(&video_path, width, height)?;

        let state = Arc::new(IdleState {
            current_frame: Arc::new(Mutex::new(Array3::zeros((height as usize, width as usize, 3)))),
            frame_id: Arc::new(AtomicU16::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            temp_file,
        });

        let state_clone = state.clone();
        thread::spawn(move || {
            state_clone.run_video_loop(actual_path, width, height);
        });

        Ok(state)
    }

    fn prepare_resource(path: &str, w: u16, h: u16) -> PyResult<(String, Option<PathBuf>)> {
        let (src_w, src_h) = Self::probe_dimensions(path).unwrap_or((0, 0));
        
        if src_w == w as u32 && src_h == h as u32 {
            return Ok((path.to_string(), None));
        }

        let temp_path = std::env::temp_dir().join(format!("idle_rescale_{}x{}.webm", w, h));
        
        let status = Command::new("ffmpeg")
            .args([
                "-y", "-i", path,
                "-vf", &format!("scale={}:{}", w, h),
                "-c:v", "libvpx-vp9",
                "-crf", "30",
                "-an",
                temp_path.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if !status.success() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("ffmpeg resize failed"));
        }

        Ok((temp_path.to_str().unwrap().to_string(), Some(temp_path)))
    }

    fn run_video_loop(&self, video_path: String, width: u16, height: u16) {
        let fps = Self::probe_fps(&video_path).unwrap_or(30.0);
        let frame_duration = Duration::from_secs_f64(1.0 / fps);

        while self.running.load(Ordering::Relaxed) {
            if let Err(e) = self.play_video_once(&video_path, width, height, frame_duration) {
                eprintln!("[IdleState] Playback error: {e}");
                thread::sleep(Duration::from_secs(1));
            }
        }
    }

    fn play_video_once(&self, path: &str, w: u16, h: u16, frame_duration: Duration) -> Result<(), String> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-i", path,
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;

        let mut stdout = child.stdout.take().ok_or("No stdout")?;
        let frame_size = (w as usize * h as usize * 3);
        let mut buf = vec![0u8; frame_size];

        while self.running.load(Ordering::Relaxed) {
            let start = Instant::now();

            if stdout.read_exact(&mut buf).is_err() { break; }

            let data = Array3::from_shape_vec((h as usize, w as usize, 3), buf.clone())
                .map_err(|e| e.to_string())?;

            {
                let mut lock = self.current_frame.lock();
                *lock = data;
            }
            self.frame_id.fetch_add(1, Ordering::Relaxed);

            let elapsed = start.elapsed();
            if elapsed < frame_duration {
                thread::sleep(frame_duration - elapsed);
            }
        }
        let _ = child.kill();
        Ok(())
    }

    fn probe_fps(video_path: &str) -> Option<f64> {
        let output = Command::new("ffprobe")
            .args([
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ])
            .output()
            .ok()?;
 
        let s = String::from_utf8_lossy(&output.stdout);
        let s = s.trim();
        let mut parts = s.splitn(2, '/');
        let num: f64 = parts.next()?.trim().parse().ok()?;
        let den: f64 = parts.next()?.trim().parse().ok()?;
        if den == 0.0 { return None; }
        Some(num / den)
    }
 
    fn probe_dimensions(video_path: &str) -> Option<(u32, u32)> {
        let output = Command::new("ffprobe")
            .args([
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path,
            ])
            .output()
            .ok()?;
 
        let s = String::from_utf8_lossy(&output.stdout);
        let s = s.trim();
        let mut parts = s.splitn(2, ',');
        let w: u32 = parts.next()?.trim().parse().ok()?;
        let h: u32 = parts.next()?.trim().parse().ok()?;
        Some((w, h))
    }
}

impl Drop for IdleState {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(ref path) = self.temp_file {
            let _ = std::fs::remove_file(path);
        }
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