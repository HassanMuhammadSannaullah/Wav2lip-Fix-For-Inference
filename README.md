# Wav2Lip-Inference-on-Python3.9
This project fixes the Wav2Lip project Inference so that it can run on Python 3.9. Wav2Lip is a project that can be used to lip-sync videos to audio. The original project was dependent on Python 3.6 and used deprecated libraries. Alot of people were unable to resolve the issues. This project fixes those problems so that Wav2Lip can now run on Python 3.9 or higher.

Original Project: https://github.com/Rudrabha/Wav2Lip

This repository enables you to perform lip-syncing using the Wav2Lip model directly in Python, offering an alternative to command-line usage. It provides a `Processor` class with methods to process video and audio inputs, generate lip-synced videos, and customize various options. You can also find the original command-line options available as arguments in this Python script.

## Getting Started

### Prerequisites

Before using this repository, ensure you have the following prerequisites installed:

- Python 3.9 or later
- Dependencies listed in `requirements.txt`

### Installing

To get started, clone this repository:

```bash
git clone https://github.com/HassanMuhammadSannaullah/Wav2lip-Fix-For-Inference.git
cd Wav2lip-Fix-For-Inference
pip install -r requirements.txt
```
## Important Note
In the decorators.py file of the librose module, make the following change to ensure compatibility:

```python
# Change this import line
from numba.decorators import jit as optional_jit

# To this
from numba import jit as optional_jit
```

## Usage
You can either directly run the wav2lip.py file in the project or import Process class from it, somewhere else in the code. Following is the sample way to run the inference

1. Import Processor class
```python
from Wav2Lip import Processor
```

2. Use run method to perform inference
```python
processor = Processor()
processor.run("path_to_face_video_or_image", "path_to_audio.wav", "output_path.mp4")
```

Additional Options
You can customize various options by providing arguments to the Processor class constructor or modifying the run method. Here are some important options:
```
# These can be set in the constructor
checkpoint_path: Path to the Wav2Lip model checkpoint. 
nosmooth: Disable smoothening of face boxes. 
static: Use a static image for face detection. 

# All below can be set in the run function of Processor class
resize_factor: Resize factor for video frames. 
rotate: Rotate frames (useful for portrait videos).
crop: Crop the video frame [y1, y2, x1, x2].
fps: Frames per second for the output video.
mel_step_size: Mel spectrogram step size.
wav2lip_batch_size: Batch size for inference.
```
For detailed information on these options, refer to the code comments in the Processor class, or refer to the original implementation of wav2lip

## Disclaimer

This project is provided for educational and entertainment purposes only. The author and contributors of this repository are not responsible for any harmful, unethical, or inappropriate use of the software or its outputs. Users are encouraged to adhere to ethical guidelines and legal regulations when using this project.

Please use this project responsibly and consider the implications of your actions. If you have any concerns or questions regarding the usage of this software, feel free to reach out for guidance.

By using this software, you agree to the above disclaimer.

## Acknowledgments
This project is built upon the Wav2Lip repository by Rudrabha Mukhopadhyay.
If you encounter any issues or have questions, feel free to open an issue 

Happy lip-syncing!


