import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
from argparse import ArgumentParser
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table as RichTable
import re
import tempfile

from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
envpaths = os.environ.get('PATH').split(';')
for envpath in envpaths:
    envpath += "\\tesseract.exe"
    if os.path.exists(envpath):
        pytesseract.pytesseract.tesseract_cmd = envpath
        break

console = Console()
verbose = None
# level 1: saved extra images
# level 2: output
# level 3: big headers
# level 4: Processing and opening/saving messages
# level 5: warnings (yellow), scores, methods and info
# Note: errors are always printed

def printer(text, lvl=None):
    if verbose == None or lvl == None:
        console.print(text)
    elif verbose >= lvl:
        console.print(text)
    

def ocr_preprocess(image):
    """
    Final OCR preprocessing:
    - 2x upscale
    - grayscale
    - gaussian blur
    - adaptive threshold (B/W)
    """
    img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    return thresh


def show_preprocessed_image(cv2_img, title="Preprocessed Image"):
    """
    Displays the final preprocessed image to the user.
    Works reliably on Windows.
    """
    if not isinstance(cv2_img, np.ndarray):
        return

    # Handle both grayscale and color images
    if len(cv2_img.shape) == 2:
        # Grayscale image
        pil_img = Image.fromarray(cv2_img)
    else:
        # Convert BGR → RGB
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    # Save temp image so OS viewer opens it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_img.save(tmp.name)
    tmp.close()

    # Open with default image viewer
    os.startfile(tmp.name)


# =========================================================
# IMPROVED IMG2TABLE EXTRACTOR
# =========================================================
class Img2TableExtractor:
    def __init__(self, language="eng"):
        self.language = language
        self.ocr_backend = TesseractOCR(lang=language)

    def extract(self, image):
        """
        Extract table using img2table library
        image: OpenCV numpy array or file path string
        returns: list of rows (list of lists of strings)
        """
        cleanup_temp = False
        image_path = None
        
        try:
            # Handle different input types
            if isinstance(image, str):
                # Already a file path
                image_path = image
            elif isinstance(image, np.ndarray):
                # NumPy array - save to temp file
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Create temporary file
                fd, image_path = tempfile.mkstemp(suffix='.png')
                os.close(fd)
                pil_img.save(image_path)
                cleanup_temp = True
            else:
                console.print(f"[red]Unsupported image type:[/red] {type(image)}")
                return []
            
            # Create img2table document from file path
            doc = Img2TableImage(src=image_path)
            
            # Extract tables
            extracted = doc.extract_tables(
                ocr=self.ocr_backend,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=40
            )
            
            if not extracted:
                return []
            
            # Get first table and convert to list
            df = extracted[0].df
            table_data = df.fillna("").astype(str).values.tolist()
            
            # Filter out empty rows
            table_data = [row for row in table_data if any(str(cell).strip() for cell in row)]
            
            return table_data
            
        except Exception as e:
            printer(f"[yellow]img2table error: {e}[/yellow]", lvl=5)
            return []
            
        finally:
            # Cleanup temporary file
            if cleanup_temp and image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except:
                    pass

# =========================================================
# SMART ORIENTATION CORRECTOR
# =========================================================
class SmartOrientationCorrector:
    COMMON_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'is', 'was', 'are', 'hello', 'name', 'yes', 'why', 'join', 'us', 'write',
        'example', 'good', 'birthday', 'invited', 'bash', 'celebrating'
    }
    
    @staticmethod
    def score_orientation(img, language="eng"):
        """Score based on REAL WORDS"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang=language, config='--oem 3 --psm 6')
        
        if not text.strip():
            return 0
        
        words = text.lower().split()
        valid_words = sum(1 for w in words if re.sub(r'[^\w]', '', w) in SmartOrientationCorrector.COMMON_WORDS)
        score = valid_words * 100 + len(words) * 5
        
        return score
    
    @staticmethod
    def fix(image, language="eng"):
        printer("[yellow]Testing orientations...[/yellow]", lvl=4)
        
        versions = {
            0: image.copy(),
            90: cv2.rotate(image.copy(), cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(image.copy(), cv2.ROTATE_180),
            270: cv2.rotate(image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        results = []
        for angle, img in versions.items():
            score = SmartOrientationCorrector.score_orientation(img, language)
            results.append((angle, img, score))
            printer(f"[dim]{angle:3d}°: score={score}[/dim]", lvl=5)
        
        best_angle, best_img, best_score = max(results, key=lambda x: x[2])
        
        if best_score < 50:
            printer("[yellow]⚠ Low confidence, keeping original[/yellow]", lvl=5)
            return image, 0
        
        printer(f"[green]✓ Selected: {best_angle}° (score: {best_score})[/green]", lvl=4)
        return best_img, best_angle


class ImprovedHandwrittenOCR:
    def __init__(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def extract(self):
        printer("[cyan]Processing handwritten text...[/cyan]", lvl=4)

        results = []
        preprocessed_images = []

        # -------------------------------
        # Method 1: Reliable 2x upscale
        # -------------------------------
        img1 = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        cv2.imwrite("handwritten_meth1_gray1.png", gray1)
        thresh1 = cv2.adaptiveThreshold(
            gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        text1 = pytesseract.image_to_string(thresh1, lang=self.language, config='--oem 1 --psm 6')
        results.append(text1.strip())
        preprocessed_images.append(thresh1)
        printer(f"[dim]Method 1: {len(text1.strip())} chars[/dim]", lvl=5)

        # -------------------------------
        # Method 2: 3x upscale + contrast
        # -------------------------------
        img2 = cv2.resize(self.image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
        gray2 = cv2.convertScaleAbs(gray2, alpha=1.3, beta=0)
        thresh2 = cv2.adaptiveThreshold(
            gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        text2 = pytesseract.image_to_string(thresh2, lang=self.language, config='--oem 1 --psm 6')
        results.append(text2.strip())
        preprocessed_images.append(thresh2)
        printer(f"[dim]Method 2: {len(text2.strip())} chars[/dim]", lvl=5)

        # -------------------------------
        # Pick the longest text (most characters)
        # -------------------------------
        best_idx = max(range(len(results)), key=lambda i: len(results[i]))
        best = results[best_idx]
        self.preprocessed_img = preprocessed_images[best_idx]
        printer(f"[green]✓ Best result: {len(best)} characters (Method {best_idx + 1})[/green]", lvl=4)

        return best


# -----------------------------
# IMPROVED TABLE EXTRACTOR
# -----------------------------
class ImprovedTableExtractor:
    def __init__(self, image):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.preprocessed_img = None

    def extract(self, num_columns=None):
        """Enhanced table extraction with proper column alignment"""
        
        # First try img2table (best for bordered tables)
        printer("[dim]Trying img2table method...[/dim]", lvl=4)
        img2table_result = Img2TableExtractor().extract(self.image)
        if img2table_result and len(img2table_result) > 0:
            printer(f"[green]✓ img2table found {len(img2table_result)} rows[/green]", lvl=4)
            self.preprocessed_img = ocr_preprocess(self.image)
            return img2table_result
        
        printer("[dim]img2table failed, using OCR-based extraction...[/dim]", lvl=5)
        
        # Fallback: OCR-based extraction with column clustering
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.preprocessed_img = ocr_preprocess(self.image)
        
        # Get word-level data
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')

        words = []
        for i, txt in enumerate(data['text']):
            txt = txt.strip()
            if txt and data['conf'][i] > 30:  # Filter low confidence
                words.append({
                    'text': txt,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'cx': data['left'][i] + data['width'][i] // 2,
                    'cy': data['top'][i] + data['height'][i] // 2
                })

        if not words:
            return []

        # Group words by rows using y-coordinate clustering
        words.sort(key=lambda w: w['y'])
        rows = []
        current_row = [words[0]]
        row_height_threshold = 20  # Pixels
        
        for w in words[1:]:
            # Check if word belongs to current row
            if abs(w['y'] - current_row[0]['y']) < row_height_threshold:
                current_row.append(w)
            else:
                # Sort current row by x position before adding
                current_row.sort(key=lambda x: x['x'])
                rows.append(current_row)
                current_row = [w]
        
        # Don't forget the last row
        if current_row:
            current_row.sort(key=lambda x: x['x'])
            rows.append(current_row)

        if not rows:
            return []

        # Determine column positions using x-coordinate clustering
        if num_columns is None:
            # Collect all x-positions
            all_x_positions = [w['cx'] for row in rows for w in row]
            all_x_positions.sort()
            
            # Find column boundaries using gaps
            column_boundaries = []
            if all_x_positions:
                column_boundaries.append(all_x_positions[0])
                
                for i in range(1, len(all_x_positions)):
                    gap = all_x_positions[i] - all_x_positions[i-1]
                    if gap > 50:  # Significant gap indicates new column
                        column_boundaries.append((all_x_positions[i-1] + all_x_positions[i]) // 2)
                
                num_columns = len(column_boundaries)
                printer(f"[dim]Auto-detected {num_columns} columns[/dim]", lvl=4)
        else:
            # Divide image width into equal columns
            img_width = self.image.shape[1]
            column_boundaries = [i * img_width // num_columns for i in range(num_columns)]

        # Build table by assigning words to columns
        table = []
        for row in rows:
            row_data = [''] * max(num_columns, 1)
            
            for word in row:
                # Find which column this word belongs to
                word_x = word['cx']
                col_idx = 0
                
                for idx, boundary in enumerate(column_boundaries):
                    if word_x >= boundary:
                        col_idx = idx
                
                # Ensure we don't exceed column count
                col_idx = min(col_idx, len(row_data) - 1)
                
                # Append word to column (handle multiple words per cell)
                if row_data[col_idx]:
                    row_data[col_idx] += ' ' + word['text']
                else:
                    row_data[col_idx] = word['text']
            
            # Only add non-empty rows
            if any(cell.strip() for cell in row_data):
                table.append(row_data)
        
        return table


# =========================================================
# TEXT EXTRACTOR
# =========================================================
class TextExtractor:
    def __init__(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def extract(self):
        img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        
        # Store preprocessed version
        self.preprocessed_img = thresh
        
        text = pytesseract.image_to_string(thresh, lang=self.language, config='--oem 3 --psm 6')
        return text.strip()


# =========================================================
# IMPROVED MIXED CONTENT HANDLER
# =========================================================
class ImprovedMixedContentHandler:
    def __init__(self, image, language="eng"):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()
        self.language = language
        self.preprocessed_img = None
    
    def detect_table_region(self):
        """Enhanced table detection using multiple signals"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Line detection
        edges = cv2.Canny(gray, 50, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=2)
        table_mask = cv2.add(h_lines, v_lines)
        
        # Method 2: Text density analysis
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--psm 6')
        
        # Create density map
        density_map = np.zeros_like(gray)
        for i, txt in enumerate(data['text']):
            if txt.strip() and data['conf'][i] > 30:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                cv2.rectangle(density_map, (x, y), (x+w, y+h), 255, -1)
        
        # Combine signals
        combined = cv2.add(table_mask, density_map // 4)
        combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            printer("[yellow]No contours detected in table region analysis[/yellow]", lvl=5)
            return None
        
        # Find largest rectangular region
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = w * h
        img_area = self.image.shape[0] * self.image.shape[1]
        
        # More lenient threshold for table detection
        area_percentage = area / img_area * 100
        printer(f"[dim]Largest region: {w}x{h} ({area_percentage:.1f}% of image)[/dim]", lvl=5)
        
        if area > img_area * 0.02:  # At least 2% of image (lowered from 3%)
            printer(f"[green]✓ Table region detected: {w}x{h} ({area_percentage:.1f}% of image)[/green]", lvl=4)
            
            # Save debug image
            debug_img = self.image.copy()
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imwrite("debug_table_region.png", debug_img)
            printer("[dim]Saved: debug_table_region.png[/dim]", lvl=1)
            
            return {'x': x, 'y': y, 'w': w, 'h': h}
        
        printer(f"[yellow]Region too small ({area_percentage:.1f}%), no table detected[/yellow]", lvl=5)
        return None
    
    def extract_mixed(self, num_columns=None):
        result = {'text_above': None, 'table': None, 'text_below': None, 'full_text': None}
        table_region = self.detect_table_region()
        
        # Store preprocessed version of full image
        self.preprocessed_img = ocr_preprocess(self.image)
        
        if table_region:
            x, y, w, h = table_region['x'], table_region['y'], table_region['w'], table_region['h']
            
            # Add padding
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(self.image.shape[1], x + w + padding)
            y_end = min(self.image.shape[0], y + h + padding)
            
            # Extract table region
            table_img = self.image[y_start:y_end, x_start:x_end]
            
            # IMPORTANT: Use ImprovedTableExtractor which has the fixed img2table logic
            printer("[dim]Extracting table from detected region...[/dim]", lvl=4)
            extractor = ImprovedTableExtractor(table_img)
            table = extractor.extract(num_columns)
            
            if table and len(table) > 0:
                result['table'] = table
                printer(f"[green]✓ Extracted table: {len(table)} rows × {len(table[0]) if table else 0} columns[/green]", lvl=4)
            else:
                printer("[yellow]⚠ Table region detected but extraction failed[/yellow]", lvl=4)
            
            # Text above table
            if y_start > 20:
                text_above_img = self.image[0:y_start, :]
                text_above = TextExtractor(text_above_img, self.language).extract()
                if text_above and len(text_above.strip()) > 0:
                    result['text_above'] = text_above
            
            # Text below table
            if y_end < self.image.shape[0] - 20:
                text_below_img = self.image[y_end:, :]
                text_below = TextExtractor(text_below_img, self.language).extract()
                if text_below and len(text_below.strip()) > 0:
                    result['text_below'] = text_below
        else:
            printer("[yellow]No table detected, treating as pure text[/yellow]", lvl=5)
            result['text_above'] = TextExtractor(self.image, self.language).extract()
        
        # Full text extraction
        result['full_text'] = TextExtractor(self.image, self.language).extract()
        return result


# =========================================================
# MAIN PROCESSOR
# =========================================================
class OCRProcessor:
    def __init__(self, image_path, language="eng"):
        self.image_path = image_path
        self.language = language
        self.final_preprocessed_img = None
    
    def process(self, mode, num_columns=None, fix_orientation=True, save_comparison=False):
        img_original = cv2.imread(self.image_path)
        if img_original is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        img = img_original.copy()
        
        # Save original for comparison
        if save_comparison:
            cv2.imwrite("original_image.png", img_original)
            printer("[dim]Saved: original_image.png[/dim]", lvl=1)
        
        if fix_orientation:
            printer("[bold cyan]═══ ORIENTATION CHECK ═══[/bold cyan]", lvl=3)
            img, angle = SmartOrientationCorrector.fix(img, self.language)
            if save_comparison and angle != 0:
                cv2.imwrite("corrected_orientation.png", img)
                printer("[dim]Saved: corrected_orientation.png[/dim]", lvl=1)
        
        printer(f"[bold cyan]═══ MODE: {mode.upper()} ═══[/bold cyan]\n", lvl=3)
        
        result = {'mode': mode, 'text': None, 'table': None, 'text_above': None, 'text_below': None}
        
        if mode == "handwritten":
            hw = ImprovedHandwrittenOCR(img, self.language)
            result['text'] = hw.extract()
            self.final_preprocessed_img = hw.preprocessed_img
            
        elif mode == "mixed":
            handler = ImprovedMixedContentHandler(img, self.language)
            mixed = handler.extract_mixed(num_columns)
            result.update(mixed)
            self.final_preprocessed_img = handler.preprocessed_img
            
        elif mode == "table":
            extractor = ImprovedTableExtractor(img)
            result['table'] = extractor.extract(num_columns)
            txt_ext = TextExtractor(img, self.language)
            result['text'] = txt_ext.extract()
            self.final_preprocessed_img = extractor.preprocessed_img
            
        else:  # text
            txt_ext = TextExtractor(img, self.language)
            result['text'] = txt_ext.extract()
            self.final_preprocessed_img = txt_ext.preprocessed_img
        
        return result


# =========================================================
# APP
# =========================================================
class OCRApp:
    LANGUAGES = {
        "1": ("English", "eng"),
        "2": ("Spanish", "spa"),
        "3": ("French", "fra"),
        "4": ("German", "deu")
    }
    
    def run(self, imgpath=None):
        printer(Panel(
            "[bold cyan]IMPROVED OCR SYSTEM - TABLE EXTRACTION FIXED[/bold cyan]\n"
            "✓ Enhanced img2table with borderless support\n"
            "✓ Improved column detection & alignment\n"
            "✓ Better mixed content detection\n"
            "✓ Multi-signal table region detection\n"
            "✓ Smart orientation (word-based)\n"
            "✓ Handwritten text support\n"
            "✓ Multi-language support\n"
            "✓ Debug visualization",
            expand=False
        ), lvl=5)
        
        while True:
            if imgpath == None:
                imgpath = Prompt.ask("\nImage path").strip('"')
            if os.path.exists(imgpath):
                printer("[green]✓ Opening image...[/green]", lvl=4)
                os.startfile(imgpath)
                break
            console.print("[bold red]File not found:[/bold red] ", imgpath)
        
        console.print("\n[bold]Languages:[/bold]")
        for k, (name, _) in self.LANGUAGES.items():
            console.print(f"  {k}. {name}")
        
        lang_choice = Prompt.ask("Language", choices=self.LANGUAGES.keys(), default="1")
        language = self.LANGUAGES[lang_choice][1]
        
        console.print("\n[bold]Modes:[/bold]")
        console.print("  1. Table")
        console.print("  2. Text")
        console.print("  3. Handwritten")
        console.print("  4. Mixed (Text + Table)")
        
        mode_choice = Prompt.ask("Mode", choices=["1", "2", "3", "4"], default="1")
        modes = {"1": "table", "2": "text", "3": "handwritten", "4": "mixed"}
        mode = modes[mode_choice]
        
        num_cols = None
        if mode in ["table", "mixed"]:
            auto = Prompt.ask("Auto-detect columns?", choices=["y", "n"], default="y")
            if auto == "n":
                num_cols = int(Prompt.ask("Number of columns", default="6"))
        
        fix_orient = Prompt.ask("Check orientation?", choices=["y", "n"], default="y") == "y"
        save_comparison = Prompt.ask("Save comparison images?", choices=["y", "n"], default="y") == "y"
        
        console.print()
        with Status("[green]Processing...[/green]"):
            processor = OCRProcessor(imgpath, language)
            result = processor.process(
                mode=mode, 
                num_columns=num_cols, 
                fix_orientation=fix_orient,
                save_comparison=save_comparison
            )
        
        console.print()
        
        # Display results
        if result['text_above']:
            printer("[yellow]📄 TEXT ABOVE TABLE:[/yellow]", lvl=2)
            printer(Panel(result['text_above'][:300], expand=False), lvl=2)
        
        if result['table']:
            display = RichTable(show_header=False, show_lines=True, title="📊 TABLE")
            
            max_cols = len(result['table'][0]) if result['table'] else 0
            for i in range(max_cols):
                display.add_column(f"Col{i+1}", style="cyan", overflow="fold", max_width=30)
            
            for row in result['table']:
                display.add_row(*row)
            
            printer(display, lvl=2)
            printer(f"[green]✓ {len(result['table'])} rows × {max_cols} columns[/green]\n", lvl=2)
        
        if result['text_below']:
            printer("[yellow]📄 TEXT BELOW TABLE:[/yellow]", lvl=2)
            printer(Panel(result['text_below'][:300], expand=False), lvl=2)
        
        if result['text'] and not result['table']:
            printer("[yellow]📄 TEXT:[/yellow]", lvl=2)
            preview = result['text'][:500] + ("..." if len(result['text']) > 500 else "")
            printer(Panel(preview, expand=False), lvl=2)
        
        # Save files
        base = os.path.splitext(os.path.basename(imgpath))[0]
        saved = []
        
        if result['table']:
            f = f"{base}_table.txt"
            with open(f, 'w', encoding='utf-8') as file:
                for row in result['table']:
                    file.write(' | '.join(row) + '\n')
            saved.append(f)
        
        if result['text_above']:
            f = f"{base}_text_above.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text_above'])
            saved.append(f)
        
        if result['text_below']:
            f = f"{base}_text_below.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text_below'])
            saved.append(f)
        
        if result['text']:
            f = f"{base}_text.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['text'])
            saved.append(f)
        
        if result.get('full_text'):
            f = f"{base}_full_text.txt"
            with open(f, 'w', encoding='utf-8') as file:
                file.write(result['full_text'])
            saved.append(f)
        
        if saved:
            printer(f"\n[green]✓ Saved: {', '.join(saved)}[/green]", lvl=4)
        
        # Show comparison images info
        if save_comparison:
            printer("\n[bold cyan]Comparison Images:[/bold cyan]", lvl=4)
            printer("  • original_image.png - Your uploaded image", lvl=4)
            if fix_orient:
                printer("  • corrected_orientation.png - After rotation (if rotated)", lvl=4)
            if mode == "mixed":
                printer("  • debug_table_region.png - Detected table region", lvl=4)
        
        # Show final preprocessed image (what OCR actually saw)
        if hasattr(processor, "final_preprocessed_img") and processor.final_preprocessed_img is not None:
            console.print("\n[bold green]Opening preprocessed image (what OCR actually read)...[/bold green]")
            show_preprocessed_image(processor.final_preprocessed_img, "Final Preprocessed Image")
        
        if saved:
            os.startfile(saved[0])

    def run_cli(self, imgpath, language="eng", mode="text", fix_orient=False, 
                    columns=None, savecompares=False):
        # language: "eng", "spa", "fra", "deu"
        # mode: "table", "text", "handwritten", "mixed"
        # columns: None is auto detect, integer otherwise
        if not os.path.exists(imgpath):
            console.print("[bold red]File not found:[/bold red] ", imgpath)
            return

        processor = OCRProcessor(imgpath, language)
        result = processor.process(
            mode=mode, 
            num_columns=columns, 
            fix_orientation=fix_orient,
            save_comparison=savecompares
        )
        return result


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Use OCR on images for text and tables.",
    )
    parser.add_argument('-i', '--image', type = str, default=None,
                        help ='the path to the image')

    parser.add_argument('-c', '--cli', action='store_true',
	                    help='command line: do not prompt for options, given via CLI')

    parser.add_argument('-d', '--folder', type = str, default=None,
                        help ='the path to a folder with images')

    parser.add_argument('-v', '--verbose', type = int, default=None,
	                    help='Verbosity level: 0=no output, 1=least,... default all')

    parser.add_argument('-l', '--language', type = str, default="eng",
                        help ='the language to parse: "eng", "spa", "fra", "deu"')
                        
    parser.add_argument('-m', '--mode', type = str, default="mixed",
                        help ='the mode: "table", "text", "handwritten", "mixed"')

    parser.add_argument('--cols', type = int, default=None,
	                    help='Check for specific columns. default auto')
                        
    parser.add_argument('-o', '--orientationcheck', action='store_true',
	                    help='Check for orientation. default false')
                        
    parser.add_argument('-s', '--savecompares', action='store_true',
	                    help='Save intermediate (comparison) images. default false')
                        
    args = parser.parse_args()

    try:
        verbose = args.verbose
        app = OCRApp()
        if args.folder == None:
            imgpaths = {args.image}
        else:
            files = os.listdir(args.folder)
            imgpaths = (args.folder + file for file in files if file.endswith(('.jpg', '.png', 'jpeg')))

        for img in imgpaths:
            if args.cli:
                result = app.run_cli(img,
                            language=args.language, 
                            mode=args.mode,
                            fix_orient=args.orientationcheck,
                            columns=args.cols,
                            savecompares=args.savecompares
                )
                printer(result, lvl=2)
            else:
                app.run(imgpath=args.image)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
