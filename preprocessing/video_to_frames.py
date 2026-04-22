import cv2
import os

VIDEO_DIR = "dataset/videos"
FRAME_DIR = "dataset/frames"
FRAME_RATE = 5   # save every 5th frame


def extract_frames_from_all_videos():
    os.makedirs(FRAME_DIR, exist_ok=True)

    videos = os.listdir(VIDEO_DIR)
    if len(videos) == 0:
        raise RuntimeError("No videos found in dataset/videos")

    frame_count = 0

    for video_name in videos:
        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_name}")
            continue

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % FRAME_RATE == 0:
                frame_name = f"frame_{frame_count}.jpg"
                cv2.imwrite(os.path.join(FRAME_DIR, frame_name), frame)
                frame_count += 1

            count += 1

        cap.release()
        print(f"[INFO] Extracted frames from {video_name}")

    print(f"[DONE] Total frames saved: {frame_count}")


if __name__ == "__main__":
    extract_frames_from_all_videos()
