import os
import csv
import json
import time
import queue
import random
import re
import threading
from datetime import datetime

import rospy
import rosservice
import rostopic
from vosk import Model, KaldiRecognizer, SetLogLevel

# experiment settings
sample_rate = 16000
frame_ms = 30
frame_bytes = 960
silence_limit = 10.0
answer_limit = 120.0
smile_duration = 1.0

# voice settings
energy_threshold = 300
voice_frames = 5
voice_needed = 3

# story text
story = (
    "Nanci was having a tea party with her friends. "
    "Everyone was having a good time. "
    "But then, Milo accidentally broke a cup. "
    "He felt very sad. "
    "Nanci saw that Milo was upset. "
    "She wanted to help him feel better. "
    "First, she helped Milo clean up the broken cup. "
    "Then, she gave him a hug and brought him a new cup."
)

# experiment questions
questions = [
    "Okay, first question. Do you like the story?",
    "Why do you think Milo felt sad when the cup was broken?",
    "Have you ever had sympathy for a friend who was sad?"
]

# robot speech
intro = (
    "Hi, I'm Q-bert! I'd like to share a short story with you "
    "and hear what you think."
)
story_intro = "Here's our story."
extra_time = "Don't worry, I'll give you more time."
explicit_next = "Okay, next question."
timeout_next = "It's all right. Next question."
normal_ending = (
    "Okay, that's the end of the experiment. "
    "Thank you for taking part!"
)
timeout_ending = (
    "It's all right. That's the end of the experiment. "
    "Thank you for taking part!"
)
stop_ending = "Of course. We can start again whenever you're ready."

# balanced orders
wait_orders = [
    [4, 7, 10], [4, 10, 7],
    [7, 4, 10], [7, 10, 4],
    [10, 4, 7], [10, 7, 4]
]

# CSV columns
columns = [
    "participant_id", "smile_condition",
    "question_number", "waiting_condition",
    "speech_detected", "response_stage", "response_latency",
    "speech_duration", "answer_duration", "follow_up_given", "follow_up_cancelled",
    "end_reason", "condition_order"
]


class StopExperiment(Exception):
    pass


def clean_text(words):
    # clean words
    words = words.lower()
    words = re.sub(r"[^a-z ]", " ", words)
    return " ".join(words.split())


def save_row(filename, row):
    # save result
    with open(filename, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)


def show_summary(filename, pid):
    # read results
    with open(filename, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # print table
    print("\nParticipant", pid, "- Results")
    print("-" * 70)
    print("{:<10} {:>18} {:>18}  {:<17}".format(
        "Question", "Before speaking", "Answer duration", "Response"
    ))
    print("-" * 70)

    for row in rows:
        if row["speech_detected"] == "True":
            before = "%.2fs" % float(row["response_latency"])
            duration = "%.2fs" % float(row["answer_duration"])
            response = "Before prompt" if row["response_stage"] == "first_wait" else "After prompt"
        else:
            before = "-"
            duration = "-"
            response = "No response"

        print("{:<10} {:>18} {:>18}  {:<17}".format(
            "Q" + row["question_number"], before, duration, response
        ))

    print("-" * 70)

    # check saved rows
    if len(rows) != 3:
        print("WARNING: Saved", len(rows), "of 3 question results")


class AudioSystem:
    def __init__(self):
        # load Vosk
        SetLogLevel(-1)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        if not os.path.isdir(model_path):
            raise RuntimeError("Vosk model folder missing: " + model_path)
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, sample_rate)

        # audio states
        self.audio_queue = queue.Queue(maxsize=500)
        self.buffer = b""
        self.stream = None
        self.subscriber = None
        self.source = ""
        self.history = []
        self.speaking = False
        self.first_voice = None
        self.last_voice = None
        self.speech_duration = 0.0
        self.last_words = ""
        self.listen_start = None
        self.threshold = energy_threshold

    def add_audio(self, data):
        # store audio
        if not data:
            return
        self.buffer += data
        while len(self.buffer) >= frame_bytes:
            frame = self.buffer[:frame_bytes]
            self.buffer = self.buffer[frame_bytes:]
            try:
                self.audio_queue.put_nowait((frame, time.monotonic()))
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait((frame, time.monotonic()))
                except queue.Empty:
                    pass

    def ros_callback(self, msg):
        # receive ROS audio
        data = getattr(msg, "data", None)
        if data is None:
            return
        if isinstance(data, bytes):
            self.add_audio(data)
        elif isinstance(data, (list, tuple)):
            import struct
            if not data:
                return
            if min(data) >= 0 and max(data) <= 255:
                self.add_audio(bytes(data))
            else:
                samples = [max(-32768, min(32767, int(x))) for x in data]
                self.add_audio(struct.pack("<" + "h" * len(samples), *samples))

    def pc_callback(self, data, frames, time_info, status):
        # receive PC audio
        self.add_audio(bytes(data))

    def start(self):
        # try ROS microphone
        topic = "/qt_respeaker_app/channel0"
        try:
            msg_class, real_topic, _ = rostopic.get_topic_class(topic, blocking=False)
            if msg_class is not None:
                self.subscriber = rospy.Subscriber(
                    real_topic, msg_class, self.ros_callback, queue_size=20
                )
                self.source = "ROS microphone: " + real_topic
                print(self.source)
                return
        except Exception as error:
            print("ROS microphone unavailable:", error)

        # try PC microphone
        try:
            import sounddevice as sd
            self.stream = sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=480,
                dtype="int16",
                channels=1,
                callback=self.pc_callback
            )
            self.stream.start()
            self.source = "Computer microphone"
            print(self.source)
        except Exception as error:
            raise RuntimeError("No working microphone: " + str(error))

    def close(self):
        # close audio
        if self.subscriber is not None:
            self.subscriber.unregister()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def reset(self):
        # clear states
        self.rec.Reset()
        self.history = []
        self.speaking = False
        self.first_voice = None
        self.last_voice = None
        self.speech_duration = 0.0
        self.last_words = ""
        self.listen_start = time.monotonic()
        self.buffer = b""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def check_words(self, data):
        # recognize commands
        complete = self.rec.AcceptWaveform(data)
        result = self.rec.Result() if complete else self.rec.PartialResult()
        words = clean_text(json.loads(result).get("text" if complete else "partial", ""))
        if words and words != self.last_words:
            print("Heard:", words)
            self.last_words = words
        if "finish the test" in words:
            raise StopExperiment()
        if "thats it" in words or "that s it" in words or "that is it" in words:
            return "explicit_end"
        return None

    def process(self):
        # process frame
        try:
            data, stamp = self.audio_queue.get(timeout=0.05)
        except queue.Empty:
            return None

        command = self.check_words(data)
        if command is not None:
            return command

        # detect voice
        import audioop
        level = audioop.rms(data, 2)
        voice = level >= self.threshold
        self.history.append(voice)
        self.history = self.history[-voice_frames:]
        if voice:
            self.last_voice = stamp
            if self.speaking:
                self.speech_duration += frame_ms / 1000.0
        if not self.speaking and len(self.history) >= voice_frames:
            if sum(self.history) >= voice_needed:
                self.speaking = True
                self.first_voice = stamp
                self.last_voice = stamp
                print("Speech started")
        return None

    def wait_for_speech(self, seconds):
        # wait for answer
        self.reset()
        start = time.monotonic()
        print("Waiting:", seconds, "seconds")
        while time.monotonic() - start < seconds:
            command = self.process()
            if command == "explicit_end":
                return None, start, "explicit_end"
            if self.first_voice is not None:
                return self.first_voice, start, "speech"
        return None, start, "no_response"

    def listen_to_answer(self):
        # finish answer
        start = time.monotonic()
        while True:
            command = self.process()
            now = time.monotonic()
            if command == "explicit_end":
                return "explicit_end"
            if now - start >= answer_limit:
                return "answer_timeout"
            if self.last_voice is not None:
                if now - self.last_voice >= silence_limit:
                    return "ten_second_silence"


class RobotVoice:
    def __init__(self, use_smile):
        # face mode
        self.use_smile = use_smile

        # connect ROS speech
        talk_name = "/qt_robot/behavior/talkText"
        rospy.wait_for_service(talk_name, timeout=15)
        talk_type = rosservice.get_service_class_by_name(talk_name)
        if talk_type is None:
            raise RuntimeError("Cannot find QTrobot talk service type")
        self.talk = rospy.ServiceProxy(talk_name, talk_type)

        # connect optional smile
        self.emotion = None
        if self.use_smile:
            emotion_name = "/qt_robot/emotion/show"
            try:
                emotion_type = rosservice.get_service_class_by_name(emotion_name)
                if emotion_type is not None:
                    self.emotion = rospy.ServiceProxy(emotion_name, emotion_type)
            except Exception as error:
                print("Smile service unavailable:", error)
        print("QTrobot speech connected")

    def smile_worker(self):
        # play smile
        try:
            self.emotion("QT/happy")
        except Exception as error:
            print("Smile skipped:", error)

    def smile(self):
        # start smile
        if self.emotion is None:
            return
        worker = threading.Thread(target=self.smile_worker, daemon=True)
        worker.start()

    def speak(self, words, audio, smile=True):
        # speak text
        print("\nQ-bert:", words, flush=True)
        if smile and self.use_smile:
            self.smile()
        audio.reset()
        result = self.talk(words)
        print("Speech service returned:", result)
        audio.reset()


def run_question(audio, robot, number, wait, pid, filename, order, smile_condition):
    # ask question
    robot.speak(questions[number - 1], audio, smile=True)
    question_end = time.monotonic()
    first, start, status = audio.wait_for_speech(wait)
    follow_up = False
    stage = "first_wait"

    # give extra time
    if first is None and status != "explicit_end":
        follow_up = True
        robot.speak(extra_time, audio, smile=True)
        first, start, status = audio.wait_for_speech(wait)
        stage = "second_wait"

    # process answer
    if first is None:
        detected = False
        latency = ""
        duration = ""
        answer_duration = ""
        reason = "explicit_end" if status == "explicit_end" else "no_response"
    else:
        detected = True
        # time from question
        latency = round(max(0.0, first - question_end), 2)
        reason = audio.listen_to_answer()
        duration = round(audio.speech_duration, 2)
        # exclude final silence
        answer_duration = round(max(0.0, audio.last_voice - first + frame_ms / 1000.0), 2)

    # save results
    row = {
        "participant_id": pid,
        "smile_condition": smile_condition,
        "question_number": number,
        "waiting_condition": wait,
        "speech_detected": detected,
        "response_stage": stage,
        "response_latency": latency,
        "speech_duration": duration,
        "answer_duration": answer_duration,
        "follow_up_given": follow_up,
        "follow_up_cancelled": detected and not follow_up,
        "end_reason": reason,
        "condition_order": "-".join(map(str, order))
    }
    save_row(filename, row)
    print("RESULT SAVED:", row)

    # next question
    if number < 3:
        response = explicit_next if reason == "explicit_end" else timeout_next
    else:
        response = normal_ending if reason == "explicit_end" else timeout_ending

    # no smile during transition
    robot.speak(response, audio, smile=False)


def main():
    # choose face mode
    print("\nQ-bert Story Experiment")
    print("Enter = QTrobot without smile")
    print("2 + Enter = QTrobot with smile")
    choice = input("\nChoose mode: ").strip()

    if choice not in ("", "2"):
        print("Invalid choice")
        return

    use_smile = (choice == "2")
    smile_condition = "with_smile" if use_smile else "without_smile"
    print("Smile enabled:", use_smile)

    # start ROS
    rospy.init_node("qbert_story_experiment", anonymous=True)
    now = datetime.now()
    pid = now.strftime("P%H%M%S")
    order = random.choice(wait_orders)
    filename = "Qbert results " + pid + " " + now.strftime("%Y%m%d %H%M%S") + ".csv"
    print("Participant:", pid)
    print("Waiting order:", order)

    # create CSV
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

    # prepare devices
    audio = None

    try:
        robot = RobotVoice(use_smile)
        audio = AudioSystem()
        audio.start()

        try:
            # tell story
            robot.speak(intro, audio, smile=True)
            robot.speak(story_intro, audio, smile=False)
            robot.speak(story, audio, smile=False)

            # ask questions
            for number in range(1, 4):
                print("\nQUESTION:", number, "WAIT:", order[number - 1])
                try:
                    run_question(
                        audio, robot, number, order[number - 1],
                        pid, filename, order, smile_condition
                    )
                except Exception as error:
                    # show question error
                    print("QUESTION FAILED:", number, error)
                    raise

            print("\nEXPERIMENT FINISHED")

        except StopExperiment:
            # emergency ending
            print("\nEXPERIMENT STOPPED")
            robot.speak(stop_ending, audio, smile=True)

    finally:
        # close devices
        if audio is not None:
            try:
                audio.close()
            except Exception as error:
                print("Audio close failed:", error)
        print("\nCSV:", filename)

        # show saved results
        try:
            show_summary(filename, pid)
        except Exception as error:
            print("Summary failed:", error)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted")
    except Exception as error:
        print("\nERROR:", error)
        raise