import os
import sys
import cv2
import time
import torch
import socket
import argparse
import threading
from collections import deque
from transformers import TextStreamer


from videollava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


class VideoCaptureThread(threading.Thread):
    def __init__(self, port, output=None, buffer_size=500):
        super().__init__(daemon=True)
        self.port = port
        self.output = output
        self._stop_event = threading.Event()

        self.frame_buffer = deque(maxlen=buffer_size)
        self.last_capture_time = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        def _add_frame(frame):
            self.frame_buffer.append((time.time(), frame))

        capture_uri = f"udpsrc port={self.port} ! application/x-rtp, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
        print(f"Connecting to pipeline: {capture_uri}")

        cap = cv2.VideoCapture(capture_uri)
        if not cap.isOpened():
            print(f"Error: cannot open pipeline:\n{capture_uri}", file=sys.stderr)
            return

        out = None
        if self.output:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output, fourcc, fps, (width, height))
            print(f"Recording to {self.output} at {width}x{height}@{fps}fps")

        print("Press 'q' to quit")
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or read error")
                    break
                if out:
                    out.write(frame)
                _add_frame(frame=frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if out:
                out.release()

    def get_latest_frames(self, fps: float, num_frames: int):
        if len(self.frame_buffer) < num_frames:
            return []

        now = self.frame_buffer[-1][0]
        interval = 1.0 / fps

        target_times = [now - i * interval for i in range(num_frames)]
        frames = []
        buffer_list = list(self.frame_buffer)

        for target_time in target_times:
            closest = min(buffer_list, key=lambda x: abs(x[0] - target_time))
            frames.append(closest[1])
        return frames


def send_udp_message(message: str, host: str, port: int, format_function=None):
    if format_function:
        message = format_function(message)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        if isinstance(message, bytes):
            sock.sendto(message, (host, port))
        else:
            sock.sendto(message.encode("utf-8"), (host, port))
        print(f"Sent: {message} to {(host, port)}")


def extract_image_features(frames, video_processor):
    video_data = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
    video_data = torch.stack(video_data, dim=1)

    video_outputs = video_processor.transform(video_data)
    image_features = torch.stack([video_outputs])

    return {"pixel_values": image_features}

def format_message(message):
    return message.replace("</s>", "")

def main(args):
    # Video-LLaVa Settings
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    _, video_processor = processor["image"], processor["video"]

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # GStreamer Settings
    video_thread = VideoCaptureThread(port=args.port, output=args.output)
    video_thread.start()

    # Socket Settings
    udp_host = os.environ.get("UDP_TARGET_HOST")
    if not udp_host:
        raise ValueError("UDP_TARGET_HOST environment variable must be set")
    udp_port = int(os.environ.get("UDP_TARGET_PORT"))
    if not udp_host:
        raise ValueError("UDP_TARGET_PORT environment variable must be set")

    if args.save_output:
        print("frame save activated")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        directory = os.path.join(args.save_output, f"frames_{timestamp}")
        os.makedirs(directory, exist_ok=True)
        frame_no = 0

    try:
        last_frame_time = time.time()
        while True:

            current_time = time.time()
            if current_time - last_frame_time >= 1 / args.fps:
                try:
                    conv = conv_templates[args.conv_mode].copy()

                    tensor = []
                    special_token = []
                    frames = video_thread.get_latest_frames(fps=args.fps, num_frames=8)
                    if not frames:
                        continue

                    if args.save_output:
                        frame_directory = os.path.join(directory, f"{frame_no}")
                        os.makedirs(frame_directory, exist_ok=True)
                        frame_no += 1
                        for i, latest_frame in enumerate(frames):
                            frame_filename = os.path.join(
                                frame_directory, f"frame_{i}.jpg"
                            )
                            cv2.imwrite(frame_filename, latest_frame)

                    video_tensor = extract_image_features(
                        frames=frames, video_processor=video_processor
                    )["pixel_values"][0].to(model.device, dtype=torch.float16)
                    special_token += [
                        DEFAULT_IMAGE_TOKEN
                    ] * model.get_video_tower().config.num_frames
                    tensor.append(video_tensor)
                    if not tensor:
                        continue
                    inp = os.environ.get("INPUT_PROMPT")
                    if not inp:
                        raise ValueError(
                            "INPUT_PROMPT environment variable must be set"
                        )
                    if getattr(model.config, "mm_use_im_start_end", False):
                        inp = (
                            "".join(
                                [
                                    DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN
                                    for i in special_token
                                ]
                            )
                            + "\n"
                            + inp
                        )
                    else:
                        inp = "".join(special_token) + "\n" + inp
                    conv.append_message(conv.roles[0], inp)
                    prompt = conv.get_prompt()

                    input_ids = (
                        tokenizer_image_token(
                            prompt=prompt,
                            tokenizer=tokenizer,
                            image_token_index=IMAGE_TOKEN_INDEX,
                            return_tensors="pt",
                        )
                        .unsqueeze(0)
                        .to(model.device)
                    )

                    stop_str = (
                        conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    )
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(
                        keywords, tokenizer, input_ids
                    )
                    streamer = TextStreamer(
                        tokenizer, skip_prompt=True, skip_special_tokens=True
                    )

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=tensor,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            streamer=streamer,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                        outputs = tokenizer.decode(
                            output_ids[0, input_ids.shape[1] :]
                        ).strip()
                        conv.messages[-1][-1] = outputs
                        outputs = outputs.lower()
                        print(f"model outputs: {outputs}")

                        send_udp_message(
                            message=outputs,
                            host=udp_host,
                            port=udp_port,
                            format_function=format_message,
                        )

                    last_frame_time = current_time

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except IndexError as e:
                    print(f"Error during generation: {e}")
                    print("Retrying...")
                    continue
    finally:
        video_thread.stop()
        video_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port number for the UDP source."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the output video file (e.g. output.mp4). If omitted, no recording.",
    )
    parser.add_argument(
        "--save_output",
        default=None,
        help="Path to save the output frame directory (e.g. /app/output). If omitted, no recording.",
    )
    parser.add_argument(
        "--fps", type=float, default=0.3, help="Frames per second for frame capture."
    )
    args = parser.parse_args()
    main(args)
