import os
import base64
import io
import time
import asyncio
import re
import pyautogui
import pyperclip
import win32gui
import win32process
import psutil
from PIL import ImageGrab
from groq import AsyncGroq

# Prevent pyautogui from throwing errors at screen edges
pyautogui.FAILSAFE = False
# Speed up pyautogui keystroke delays
pyautogui.PAUSE = 0.03


class VisionService:
    """
    Hybrid Vision Engine v2.
    
    Two capabilities:
    1. describe_screen()  → Llama 4 Scout VLM for visual understanding (graphs, UI, images)
    2. extract_text()     → Clipboard-first extraction with VLM OCR fallback
    """

    # ================================================================
    #  DYNAMIC APP DETECTION (Pattern-Based)
    # ================================================================

    # Tier 1: Exact process name matches (fast lookup)
    _KNOWN_PROCESSES = {
        # Code Editors
        "Code.exe":               "code_editor",
        "code.exe":               "code_editor",
        "devenv.exe":             "code_editor",
        # Terminals
        "WindowsTerminal.exe":    "terminal",
        "cmd.exe":                "terminal",
        "powershell.exe":         "terminal",
        "pwsh.exe":               "terminal",
        # Browsers
        "chrome.exe":             "browser",
        "msedge.exe":             "browser",
        "firefox.exe":            "browser",
        # Documents
        "notepad.exe":            "document",
        "WINWORD.EXE":            "document",
        "EXCEL.EXE":              "document",
        "POWERPNT.EXE":           "document",
    }

    # Tier 2: Process name substring patterns (catches ANY IDE/browser/terminal)
    _PROCESS_PATTERNS = [
        # Code Editors / IDEs — match substrings in process name
        (re.compile(r"code|studio|pycharm|idea|intellij|webstorm|phpstorm|rider|"
                     r"sublime|notepad\+\+|atom|eclipse|netbeans|cursor|antigravity|"
                     r"dev[-_]?c|codeblocks|clion|goland|rubymine|fleet|zed|nova|"
                     r"brackets|komodo|geany|lite[-_]?xl|helix|lapce",
                     re.IGNORECASE), "code_editor"),
        # Terminals
        (re.compile(r"terminal|console|conemu|hyper|alacritty|kitty|wezterm|"
                     r"mintty|mobaxterm|putty|xterm|iterm|tabby|warp|shell|"
                     r"gitbash|cmder",
                     re.IGNORECASE), "terminal"),
        # Browsers
        (re.compile(r"chrome|firefox|edge|brave|opera|vivaldi|safari|arc|"
                     r"chromium|waterfox|librewolf|tor|midori|silk|maxthon",
                     re.IGNORECASE), "browser"),
        # Document/Text editors
        (re.compile(r"word|excel|powerpoint|obsidian|notion|typora|marktext|"
                     r"onenote|acrobat|foxit|sumatrapdf|okular|evince|"
                     r"libreoffice|openoffice|wps|pages|numbers|keynote",
                     re.IGNORECASE), "document"),
    ]

    # Tier 3: Window title patterns (catches apps by what they're showing)
    _TITLE_PATTERNS = [
        # Code files: .py, .js, .ts, .cpp, .java, .go, .rs, .rb, etc.
        (re.compile(r"\.(py|js|jsx|ts|tsx|cpp|c|h|hpp|cs|java|go|rs|rb|php|"
                     r"swift|kt|scala|lua|r|m|mm|sql|sh|bash|zsh|ps1|bat|"
                     r"yaml|yml|json|xml|html|css|scss|sass|less|vue|svelte|"
                     r"dart|zig|nim|ex|exs|erl|hs|ml|clj|v|vhd|asm|toml|ini|"
                     r"cfg|conf|env|dockerfile|makefile|cmake)\b",
                     re.IGNORECASE), "code_editor"),
        # Terminal indicators in title
        (re.compile(r"powershell|cmd\.exe|command prompt|bash|zsh|terminal|"
                     r"ubuntu|wsl|ssh|PS\s+[A-Z]:\\|C:\\>|~\$|λ|❯",
                     re.IGNORECASE), "terminal"),
        # Browser indicators in title
        (re.compile(r"- google chrome|- mozilla firefox|- microsoft edge|"
                     r"- brave|- opera|- vivaldi|- safari|localhost:\d+|"
                     r"https?://",
                     re.IGNORECASE), "browser"),
        # Document indicators
        (re.compile(r"\.(docx?|xlsx?|pptx?|pdf|txt|md|rtf|odt|ods|odp|csv|"
                     r"log|tex)\b",
                     re.IGNORECASE), "document"),
    ]

    def __init__(self, vision_model="meta-llama/llama-4-scout-17b-16e-instruct"):
        print(f">> [VISION] Initializing Hybrid Vision Engine v2...")
        print(f"   - VLM Engine: Groq Llama 4 Scout (17B MoE)")
        print(f"   - Text Engine: Clipboard Extraction (pyautogui)")
        print(f"   - Fallback: VLM OCR via Llama 4 Scout")

        self.model = vision_model
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    # ================================================================
    #  PUBLIC API
    # ================================================================

    async def describe_screen(self, custom_prompt=None):
        """
        Uses the Visual LLM to look at graphs, UI, and general screen state.
        Powered by Llama 4 Scout — natively multimodal, fast on Groq.
        """
        base64_image = await asyncio.to_thread(self._capture_and_encode)
        user_question = custom_prompt if custom_prompt else "Describe what is on this screen."

        final_prompt = (
            "You are the visual cortex of a human-like voice assistant. "
            f"USER'S REQUEST: '{user_question}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer based ONLY on the provided screen image.\n"
            "2. Keep it natural, casual, and limited to 1 or 2 short sentences.\n"
            "3. DO NOT use markdown formatting. No asterisks, bolding, or lists."
        )

        try:
            content_payload = [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]

            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.2,
                max_completion_tokens=150
            )

            description = response.choices[0].message.content.strip().replace("*", "")
            return description

        except Exception as e:
            print(f">> [VLM Error]: {e}")
            return "I lost my connection to the visual cortex."

    async def extract_text(self):
        """
        Smart text extraction with 3-tier fallback strategy:
        
        Tier 1: Clipboard (pyautogui Ctrl+A → Ctrl+C) — ~50ms
        Tier 2: VLM OCR (Llama 4 Scout reads the screenshot) — ~1-2s
        Tier 3: Empty string (complete failure)
        """
        # --- TIER 1: Clipboard Extraction ---
        try:
            window_info = await asyncio.to_thread(self._get_active_window)
            window_title = window_info["title"]
            process_name = window_info["process"]
            app_type = self._classify_app(process_name, window_title)

            print(f">> [VISION] Active Window: '{window_title}' ({process_name}) → {app_type}")

            # Only attempt clipboard extraction for detected app types
            if app_type != "unknown":
                extracted = await asyncio.to_thread(self._clipboard_extract, app_type)

                if extracted and len(extracted.strip()) > 10:
                    print(f">> [VISION] ✅ Clipboard extracted {len(extracted)} chars")
                    return extracted[:15000]  # Cap context window
                else:
                    print(f">> [VISION] ⚠️ Clipboard returned empty/short, falling back to VLM OCR")
            else:
                print(f">> [VISION] ⚠️ Unknown app '{process_name}', falling back to VLM OCR")

        except Exception as e:
            print(f">> [VISION] ⚠️ Clipboard extraction failed: {e}")

        # --- TIER 2: VLM OCR Fallback ---
        try:
            print(f">> [VISION] Falling back to VLM OCR (Llama 4 Scout)...")
            text = await self._vlm_ocr()
            if text:
                return text[:15000]
        except Exception as e:
            print(f">> [VISION] ❌ VLM OCR also failed: {e}")

        # --- TIER 3: Total failure ---
        return "No readable text or code found on the screen."

    # ================================================================
    #  PRIVATE: Dynamic App Classification
    # ================================================================

    @classmethod
    def _classify_app(cls, process_name: str, window_title: str) -> str:
        """
        3-tier dynamic detection:
        1. Exact process name match (fast dict lookup)
        2. Process name pattern match (regex on process name)
        3. Window title pattern match (regex on title bar text)
        
        This catches any IDE, browser, terminal, or document editor — 
        known or unknown — without needing to hardcode every process name.
        """
        # Tier 1: Exact match
        if process_name in cls._KNOWN_PROCESSES:
            return cls._KNOWN_PROCESSES[process_name]

        # Tier 2: Process name substring pattern
        for pattern, app_type in cls._PROCESS_PATTERNS:
            if pattern.search(process_name):
                return app_type

        # Tier 3: Window title pattern (catches everything else)
        for pattern, app_type in cls._TITLE_PATTERNS:
            if pattern.search(window_title):
                return app_type

        return "unknown"

    # ================================================================
    #  PRIVATE: Screenshot Capture
    # ================================================================

    def _capture_and_encode(self):
        """Captures and base64 encodes for the VLM. Optimized for speed."""
        screenshot = ImageGrab.grab()
        # Smaller resolution = faster upload, Llama 4 Scout handles it fine
        screenshot.thumbnail((960, 540))
        buffered = io.BytesIO()
        # Lower quality = smaller base64 payload = less network latency
        screenshot.save(buffered, format="JPEG", quality=60)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # ================================================================
    #  PRIVATE: Active Window Detection
    # ================================================================

    def _get_active_window(self):
        """
        Detects the currently focused foreground window.
        Returns: {"title": str, "process": str}
        """
        try:
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                process_name = "Unknown"

            return {"title": window_title, "process": process_name}

        except Exception as e:
            print(f">> [Window Detection Error]: {e}")
            return {"title": "Unknown", "process": "Unknown"}

    # ================================================================
    #  PRIVATE: Clipboard Extraction (THE FAST PATH)
    # ================================================================

    def _clipboard_extract(self, app_type: str) -> str:
        """
        Uses pyautogui to Ctrl+A → Ctrl+C → read clipboard.
        Saves and restores the original clipboard content.
        
        ~50-100ms total execution time.
        """
        original_clipboard = ""

        try:
            # 1. Save the user's current clipboard
            try:
                original_clipboard = pyperclip.paste()
            except Exception:
                original_clipboard = ""

            # 2. Clear clipboard so we can detect if copy worked
            try:
                pyperclip.copy("")
            except Exception:
                pass

            # 3. Execute the copy strategy based on app type
            if app_type == "terminal":
                # Terminals often need a different select-all
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'c')
                time.sleep(0.08)
            else:
                # Universal: Code editors, browsers, documents
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.05)
                pyautogui.hotkey('ctrl', 'c')
                time.sleep(0.08)

            # 4. Read the extracted text
            extracted_text = pyperclip.paste()

            # 5. Deselect (press Home to clear selection without moving content)
            pyautogui.press('home')

            # 6. Restore original clipboard
            if original_clipboard:
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    pass

            return extracted_text if extracted_text else ""

        except Exception as e:
            print(f">> [Clipboard Error]: {e}")
            # Always try to restore clipboard on error
            if original_clipboard:
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    pass
            return ""

    # ================================================================
    #  PRIVATE: VLM OCR Fallback
    # ================================================================

    async def _vlm_ocr(self):
        """
        Uses Llama 4 Scout to read text/code from a screenshot.
        Smarter than traditional OCR — understands code structure, layout, context.
        """
        base64_image = await asyncio.to_thread(self._capture_and_encode)

        ocr_prompt = (
            "You are an expert OCR system. Extract ALL visible text and code from this screenshot. "
            "Rules:\n"
            "1. Reproduce the text EXACTLY as shown — preserve indentation, line breaks, and formatting.\n"
            "2. If it's code, maintain the exact code structure and syntax.\n"
            "3. If it's a terminal/console, include the command output as-is.\n"
            "4. Do NOT add any commentary, headers, or descriptions. Just the raw text.\n"
            "5. If no text is visible, respond with 'NO_TEXT_FOUND'."
        )

        try:
            content_payload = [
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]

            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.0,
                max_completion_tokens=4096
            )

            text = response.choices[0].message.content.strip()

            if text == "NO_TEXT_FOUND":
                return ""

            return text

        except Exception as e:
            print(f">> [VLM OCR Error]: {e}")
            return ""