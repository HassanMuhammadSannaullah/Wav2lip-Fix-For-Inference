import os
import cv2
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from models import Wav2Lip
import audio
from datetime import datetime
import shutil


class Processor:
    def __init__(
        self,
        checkpoint_path=os.path.join(
            "checkpoints", "wav2lip_gan.pth"
        ),
        nosmooth=False,
        static=False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.static = static
        self.nosmooth = nosmooth

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T :]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        print("Detecting Faces")
        # Load the pre-trained Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(
            os.path.join(
                "checkpoints",
                "haarcascade_frontalface_default.xml",
            )
        )  # cv2.data.haarcascades
        pads = [0, 10, 0, 0]
        results = []
        pady1, pady2, padx1, padx2 = pads

        for image in images:
            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Get the first detected face (you can modify this to handle multiple faces)
                x, y, w, h = faces[0]

                # Calculate the bounding box coordinates
                x1 = max(0, x - padx1)
                x2 = min(image.shape[1], x + w + padx2)
                y1 = max(0, y - pady1)
                y2 = min(image.shape[0], y + h + pady2)

                results.append([x1, y1, x2, y2])
            else:
                cv2.imwrite(
                    os.path.join("temp","faulty_frame.jpg"), image
                )  # Save the frame where the face was not detected.
                raise ValueError("Face not detected! Ensure the image contains a face.")

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, 5)
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        return results

    def datagen(self, frames, mels):
        img_size = 96
        box = [-1, -1, -1, -1]
        wav2lip_batch_size = 128
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(
                    frames
                )  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print("Using the specified bounding box instead of face detection...")
            y1, y2, x1, x2 = box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (img_size, img_size))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, img_size // 2 :] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1],
                )

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _load(self, checkpoint_path):
        if self.device == "cuda":
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace("module.", "")] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def run(
        self,
        face,
        audio_file,
        output_path="output.mp4",
        resize_factor=4,
        rotate=False,
        crop=[0, -1, 0, -1],
        fps=25,
        mel_step_size=16,
        wav2lip_batch_size=128,
    ):
        if not os.path.isfile(face):
            raise ValueError("--face argument must be a valid path to video/image file")

        elif face.split(".")[1] in ["jpg", "png", "jpeg"]:
            full_frames = [cv2.imread(face)]
            fps = fps

        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print("Reading video frames...")

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (
                            frame.shape[1] // resize_factor,
                            frame.shape[0] // resize_factor,
                        ),
                    )

                if rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_file.endswith(".wav"):
            print("Extracting raw audio...")
            command = "ffmpeg -y -i {} -strict -2 {}".format(
                audio_file, f"{os.path.join('temp','temp.wav')}"
            )

            subprocess.call(command, shell=True)
            audio_file = os.path.join("temp", "temp.wav")

        wav = audio.load_wav(audio_file, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
            )

        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[: len(mel_chunks)]

        print("Full Frames before gen : ", len(full_frames))

        batch_size = wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
        ):
            if i == 0:
                model = self.load_model(self.checkpoint_path)
                print("Model loaded")
                generated_temp_video_path = os.path.join(
                    "temp",
                    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_result.avi",
                )
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(
                    generated_temp_video_path,
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps,
                    (frame_w, frame_h),
                )

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
                self.device
            )
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
                self.device
            )

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        # Load the video and audio clips
        video_clip = VideoFileClip(generated_temp_video_path)
        audio_clip = AudioFileClip(audio_file)

        # Set the audio of the video clip to the loaded audio clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the combined video to a new file
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    processor = Processor()
    processor.run("image_path", "audio_path")
