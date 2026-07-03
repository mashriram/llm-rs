pub mod candle;
pub mod vision;
pub mod audio;

use std::sync::Mutex;
pub static ACTIVE_IMAGE_PATH: Mutex<Option<String>> = Mutex::new(None);
pub static ACTIVE_AUDIO_PATH: Mutex<Option<String>> = Mutex::new(None);

