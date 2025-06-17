# -*- coding: utf-8 -*-

import os
import csv
import logging
from datetime import datetime
from flox import Flox  # noqa: E402
import webbrowser  # noqa: E402
import requests  # noqa: E402
import json  # noqa: E402
import pyperclip  # noqa: E402
from typing import Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import difflib
import asyncio
from edge_tts import Communicate
from playsound import playsound


PROXIES = {
    "http": os.environ.get("HTTP_PROXY", ""),
    "https": os.environ.get("HTTPS_PROXY", ""),
}

async def generate_tts_audio(text: str, filename: str = "output.mp3"):
    communicate = Communicate(text, "en-GB-LibbyNeural")
    await communicate.save(filename)



def lcs_diff_align(a: str, b: str):
    sm = difflib.SequenceMatcher(None, a, b)
    lcs = []

    a_aligned = []
    b_aligned = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            a_aligned.append(a[i1:i2])
            b_aligned.append(b[j1:j2])
        elif tag == "delete":
            a_aligned.append(a[i1:i2])
            b_aligned.append("  " * (i2 - i1))
        elif tag == "insert":
            a_aligned.append("  " * (j2 - j1))
            b_aligned.append(b[j1:j2])
        elif tag == "replace":
            a_seg = f"({a[i1:i2]})"
            b_seg = f"({b[j1:j2]})"
            len_a = len(a_seg)
            len_b = len(b_seg)
            max_len = max(len_a, len_b)

            # 计算左右补空格数
            # 总共补 max_len - 当前长度 个空格，均分左右，多余放右侧
            def pad_center(s, target_len):
                diff = target_len - len(s)
                left_pad = diff // 2
                right_pad = diff - left_pad
                return "  " * left_pad + s + "  " * right_pad

            a_aligned.append(pad_center(a_seg, max_len))
            b_aligned.append(pad_center(b_seg, max_len))

    result = ''
    result += "".join(a_aligned)
    result += '\n'
    result += "".join(b_aligned)
    return result

class Gemini(Flox):
    def __init__(self):
        self.api_key = self.settings.get("api_key")
        self.model = self.settings.get("model")
        self.prompt_stop = self.settings.get("prompt_stop")
        self.default_system_prompt = self.settings.get("default_prompt")
        self.save_conversation_setting = self.settings.get("save_conversation")
        self.log_level = self.settings.get("log_level")
        self.logger_level(self.log_level)

        try:
            self.csv_file = open("system_messages.csv", encoding="utf-8", mode="r")
            reader = csv.DictReader(self.csv_file, delimiter=";")
            self.prompts = list(reader)
            [logging.debug(f"Found prompt: {row}") for row in self.prompts]

        except FileNotFoundError:
            self.prompts = None
            logging.error("Unable to open system_messages.csv")

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
        )

    def play_audio(self, filename: str = "output.mp3") -> None:
        try:
            playsound(filename, block=True)
            logging.debug(f"played successfully")

        except Exception as e:
            logging.error(f"Error playing audio silently: {e}")


    def query(self, query: str) -> None:
        if not self.api_key:
            self.add_item(
                title="Unable to load the API key",
                subtitle=(
                    "Please make sure you've added a valid API key in the settings"
                ),
            )
            return
        if self.prompts is None:
            self.add_item(
                title="Unable to load the system prompts from CSV",
                subtitle="Please validate that the plugins folder contains a valid system_prompts.csv",  # noqa: E501
                method=self.open_plugin_folder,
            )
            return
        if query.endswith(self.prompt_stop):
            prompt, prompt_keyword, system_message = self.split_prompt(query)

            prompt_timestamp = datetime.now()
            result = ""
            # answer_timestamp = prompt_timestamp
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt),
            ]
            messages_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages])
            logging.debug(f"Sending request with data: {messages_str}")
            try:
                response = self.llm.invoke(messages)
                result = response.content if hasattr(response, "content") else str(response)
                self.add_item(
                    title=result,
                    subtitle=lcs_diff_align(query,result),
                    method=self.copy_answer,
                    parameters=[result],
                )
                answer_timestamp = datetime.now()

            except Exception as e:
                logging.error(f"Gemini API error during stream: {e}")
                self.add_item(
                    title="Gemini API Error",
                    subtitle=str(e),
                )
                return

            # 调用 edge-tts 生成音频
            try:
                asyncio.run(generate_tts_audio(result, "output.mp3"))
            except Exception as e:
                logging.error(f"TTS Generation Error: {e}")

            self.add_item(
                title="🔊",
                subtitle="",
                method=self.play_audio,
                parameters=["output.mp3"],
            )

            filename = None
            if self.save_conversation_setting:
                filename = self.save_conversation(
                    prompt_keyword, prompt, prompt_timestamp, result, answer_timestamp
                )

        else:
            self.add_item(
                title=f"Type your prompt and end with {self.prompt_stop}",
                subtitle=f"Current model: {self.model}",
            )
        return

    def save_conversation(
        self,
        keyword: str,
        prompt: str,
        prompt_timestamp: datetime,
        answer: str,
        answer_timestamp: datetime,
    ) -> str:
        filename = f"Conversations '{keyword}' keyword.txt"
        formatted_prompt_timestamp = prompt_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        formatted_answer_timestamp = answer_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        new_content = f"[{formatted_prompt_timestamp}] User: {prompt}\n[{formatted_answer_timestamp}] Gemini: {answer}\n\n"  # noqa: E501

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    existing_content = file.read()
            except PermissionError:
                logging.error(PermissionError)
        else:
            existing_content = ""

        new_content = new_content + existing_content

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(new_content)
        except PermissionError:
            logging.error(PermissionError)

        return filename

    def split_prompt(self, query: str) -> Tuple[str, str, str]:
        prompt = query.rstrip(self.prompt_stop).strip()
        prompt_array = prompt.split(" ")
        prompt_keyword = prompt_array[0].lower()

        system_message = ""

        for row in self.prompts:
            if row["Key Word"] == prompt_keyword:
                system_message = row["System Message"]
                prompt = prompt.split(" ", 1)[1]

        if not system_message:
            prompt_keyword = self.default_system_prompt

            for row in self.prompts:
                if row["Key Word"] == self.default_system_prompt:
                    system_message = row["System Message"]

        if len(prompt_array) == 1:
            prompt = prompt_array[0]

        logging.debug(
            f"""
        Prompt: {prompt}
        Prompt keyword: {prompt_keyword}
        System message: {system_message}
        """
        )

        return prompt, prompt_keyword, system_message

    def ellipsis(self, string: str, length: int):
        string = string.split("\n", 1)[0]
        return string[: length - 3] + "..." if len(string) > length else string

    def copy_answer(self, answer: str) -> None:
        """
        Copy answer to the clipboard.
        """
        pyperclip.copy(answer)

    def open_in_editor(self, filename: Optional[str], answer: Optional[str]) -> None:
        """
        Open the answer in the default text editor. If no filename is given,
        the conversation will be written to a new text file and opened.
        """
        if filename:
            webbrowser.open(filename)
            return

        if answer:
            temp_file = "temp_text.txt"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(answer)
            webbrowser.open(temp_file)
            return

    def display_answer(self, answer: str) -> None:
        """
        Display the answer directly.
        """
        self.add_item(
            title="Answer",
            subtitle=answer,
        )

    def open_plugin_folder(self) -> None:
        webbrowser.open(os.getcwd())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.csv_file.close()


if __name__ == "__main__":
    Gemini()
