from ast import arg
import os
import sys
import cv2
import json
import time
import torch
import socket
import datetime
import argparse
import threading
from collections import deque
from dotenv import load_dotenv
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
    def __init__(self, port, output=None, buffer_size=500, fps=30):
        super().__init__(daemon=True)
        self.port = port
        self.output = output
        self._stop_event = threading.Event()
        self.fps = fps
        self.frame_buffer = deque(maxlen=buffer_size)

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
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output, fourcc, fps, (width, height))
            print(f"Recording to {self.output} at {width}x{height}@{fps}fps")

        print("Press 'q' to quit")
        try:
            last_frame_time = time.time()
            while not self._stop_event.is_set():
                current_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or read error")
                    return
                if self.output and out is not None:
                    out.write(frame)
                if current_time - last_frame_time >= 1 / self.fps:
                    _add_frame(frame=frame)
                    print(f"Captured frame at {current_time:.2f} seconds")
                    last_frame_time = current_time

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if out:
                out.release()

    def get_latest_frames(self, num_frames: int):
        buffer_list = list(self.frame_buffer)

        if len(buffer_list) < num_frames:
            return []

        return [frame[1] for frame in buffer_list[-num_frames:]]


class UDPMessageSender:
    def __init__(self, host: str, port: int, max_messages: int = 100):
        self.host = host
        self.port = port
        self.messages = deque(maxlen=max_messages)

    def send_message(self, message: str, format_function=None, history_num=None):
        if format_function:
            message = format_function(message)
        messages_to_send = []
        if history_num is not None:
            messages_to_send.extend(self.get_sent_messages(count=history_num - 1))
        messages_to_send.append(message)
        full_message = "\n".join(messages_to_send)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(full_message.encode("utf-8"), (self.host, self.port))
            header = f"Sending message to {self.host}:{self.port}"
            min_width = 40
            calculated_width = len(header) + 6
            width = max(min_width, calculated_width)

            print("=" * width)
            print(header.center(width))
            print("-" * width)

            for line in messages_to_send:
                print(line.center(width))

            print("=" * width + "\n")

            self.messages.append(message)

    def get_sent_messages(self, count: int = None):
        if count is None or len(self.messages) < count:
            return list(self.messages)
        else:
            return list(self.messages)[-count:]


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
    return message.replace("</s>", "").replace("\n", "").replace("\\", "")


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
    video_thread = VideoCaptureThread(port=args.port, output=args.output, fps=args.fps)
    video_thread.start()

    # Socket Settings
    udp_host = os.environ.get("UDP_TARGET_HOST")
    if not udp_host:
        raise ValueError("UDP_TARGET_HOST environment variable must be set")
    udp_port = int(os.environ.get("UDP_TARGET_PORT"))
    if not udp_host:
        raise ValueError("UDP_TARGET_PORT environment variable must be set")
    upd_sender = UDPMessageSender(host=udp_host, port=udp_port, max_messages=5)

    if args.save_output:
        print("\n[INFO]: OUTPUT SAVE ACTIVATED")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_directory = os.path.join(args.save_output, f"frames_{timestamp}")
        os.makedirs(save_directory, exist_ok=True)
        frame_no = 0
        all_llm_data = []

    try:
        while True:
            try:
                process_start_time = time.time()
                conv = conv_templates[args.conv_mode].copy()

                tensor = []
                special_token = []
                frames = video_thread.get_latest_frames(num_frames=8)
                if not frames:
                    continue

                video_tensor = extract_image_features(frames=frames, video_processor=video_processor)["pixel_values"][
                    0
                ].to(model.device, dtype=torch.float16)
                special_token += [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames
                tensor.append(video_tensor)
                if not tensor:
                    continue
                inp = args.prompt if args.prompt else os.environ.get("INPUT_PROMPT")
                if not inp:
                    raise ValueError("INPUT_PROMPT environment variable must be set")
                if getattr(model.config, "mm_use_im_start_end", False):
                    inp = (
                        "".join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + "\n" + inp
                    )
                else:
                    inp = "".join(special_token) + "\n" + inp
                conv.append_message(conv.roles[0], inp)
                prompt = conv.get_prompt()
                print(f"prompt:\n{prompt}")

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

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=tensor,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        # streamer=streamer, #if you want to stream the output, uncomment this
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
                    conv.messages[-1][-1] = outputs
                    outputs = outputs.lower()
                    print("------------- MODEL OUTPUTS -------------")
                    print(outputs)
                    print("-----------------------------------------")

                    upd_sender.send_message(message=outputs, format_function=format_message, history_num=5)
                    process_finish_time = time.time()

                    if args.save_output:
                        frame_directory = os.path.join(save_directory, f"{frame_no}")
                        os.makedirs(frame_directory, exist_ok=True)
                        frame_no += 1
                        for i, latest_frame in enumerate(frames):
                            frame_filename = os.path.join(frame_directory, f"frame_{i}.jpg")
                            cv2.imwrite(frame_filename, latest_frame)
                        llm_output_path = os.path.join(frame_directory, "llm_output.json")
                        nine_hours_in_seconds = 9 * 3600
                        frame_received_time_human = datetime.datetime.fromtimestamp(
                            process_start_time + nine_hours_in_seconds
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        llm_reasoning_time_human = datetime.datetime.fromtimestamp(
                            time.time() + nine_hours_in_seconds
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        llm_data = {
                            "frame_no": frame_no,
                            "frame_recieved_time": frame_received_time_human,
                            "llm_reasoning_time": llm_reasoning_time_human,
                            "llm_reasoning_time_diff": process_start_time - process_finish_time,
                            "llm_input": prompt,
                            "llm_output": outputs,
                            "sent_messages": upd_sender.get_sent_messages(count=5),
                        }
                        all_llm_data.append(llm_data)
                        with open(llm_output_path, "w") as json_file:
                            json.dump(llm_data, json_file, ensure_ascii=False, indent=4)

                    process_start_time = time.time()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except IndexError as e:
                print(f"\nError during generation: {e}")
                print("Retrying...")
                continue
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                print(f"\nError during generation: {e}")
                print("Retrying...")
                continue

    except KeyboardInterrupt:
        print("\nProgram interrupted. Cleaning up...")
    finally:
        video_thread.stop()
        video_thread.join()
        if args.save_output:
            with open(os.path.join(save_directory, "all_llm_data.json"), "w") as json_file:
                json.dump(all_llm_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", type=int, default=7860, help="Port number for the UDP source.")
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
    parser.add_argument("--fps", type=float, default=0.3, help="Frames per second for frame capture.")
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Input prompt for the model.",
    )
    args = parser.parse_args()
    load_dotenv()
    main(args)
