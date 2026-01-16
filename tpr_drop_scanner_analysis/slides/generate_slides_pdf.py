# ============================================================================
# NotebookLM-Style Slide Deck Generator - Markdown to HTML/PDF
# ============================================================================
# This script converts the NotebookLM-style markdown slides to a beautiful
# HTML presentation that can be opened in a browser and printed/saved as PDF.
#
# Usage:
#   python generate_slides_pdf.py
#
# Output:
#   - notebooklm_slides.html (open in browser, print to PDF)
# ============================================================================

import os
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

INPUT_FILE = Path(__file__).parent / "notebooklm_style_slides.md"
OUTPUT_HTML = Path(__file__).parent / "notebooklm_slides.html"

# NotebookLM-inspired color scheme
COLORS = {
    'primary': '#1a73e8',      # Google Blue
    'secondary': '#34a853',     # Google Green
    'accent': '#ea4335',        # Google Red
    'warning': '#fbbc04',       # Google Yellow
    'background': '#ffffff',
    'text': '#202124',
    'light_bg': '#f8f9fa',
    'border': '#dadce0',
}

# ----------------------------------------------------------------------------
# HTML TEMPLATE
# ----------------------------------------------------------------------------

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Roboto', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .slide-container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        .slide {{
            background: white;
            border-radius: 16px;
            padding: 48px 56px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            page-break-after: always;
            min-height: 600px;
            position: relative;
        }}
        
        .slide:last-child {{
            page-break-after: avoid;
        }}
        
        .slide-number {{
            position: absolute;
            top: 20px;
            right: 24px;
            background: {primary};
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }}
        
        h1 {{
            font-family: 'Google Sans', sans-serif;
            font-size: 36px;
            color: {primary};
            margin-bottom: 24px;
            font-weight: 500;
            border-bottom: 3px solid {primary};
            padding-bottom: 16px;
        }}
        
        h2 {{
            font-family: 'Google Sans', sans-serif;
            font-size: 28px;
            color: {text};
            margin-bottom: 20px;
            font-weight: 500;
        }}
        
        h3 {{
            font-size: 20px;
            color: {primary};
            margin: 24px 0 12px 0;
            font-weight: 500;
        }}
        
        p {{
            font-size: 18px;
            line-height: 1.7;
            color: {text};
            margin-bottom: 16px;
        }}
        
        ul, ol {{
            margin: 16px 0 16px 24px;
            font-size: 17px;
            line-height: 1.8;
            color: {text};
        }}
        
        li {{
            margin-bottom: 10px;
        }}
        
        li::marker {{
            color: {primary};
        }}
        
        code {{
            background: {light_bg};
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Roboto Mono', monospace;
            font-size: 15px;
            color: {accent};
        }}
        
        pre {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 20px 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 15px;
        }}
        
        th {{
            background: {primary};
            color: white;
            padding: 14px 16px;
            text-align: left;
            font-weight: 500;
        }}
        
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid {border};
        }}
        
        tr:nth-child(even) {{
            background: {light_bg};
        }}
        
        blockquote {{
            border-left: 4px solid {primary};
            padding: 16px 24px;
            margin: 20px 0;
            background: {light_bg};
            border-radius: 0 8px 8px 0;
            font-style: italic;
            font-size: 18px;
        }}
        
        .title-slide {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            min-height: 550px;
        }}
        
        .title-slide h1 {{
            font-size: 48px;
            border: none;
            margin-bottom: 16px;
        }}
        
        hr {{
            border: none;
            height: 2px;
            background: {border};
            margin: 32px 0;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .slide {{
                box-shadow: none;
                border: 1px solid {border};
                margin: 0;
                page-break-after: always;
            }}
            
            .slide-number {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="slide-container">
        {slides_content}
    </div>
    
    <script>
        // Add slide numbers
        document.querySelectorAll('.slide').forEach((slide, index) => {{
            const num = document.createElement('div');
            num.className = 'slide-number';
            num.textContent = (index + 1);
            slide.appendChild(num);
        }});
    </script>
</body>
</html>
'''

# ----------------------------------------------------------------------------
# MARKDOWN PARSER
# ----------------------------------------------------------------------------

def parse_markdown_to_slides(markdown_content: str) -> list:
    """Parse markdown content into individual slides."""
    slides = []
    current_slide = []
    
    lines = markdown_content.split('\n')
    
    for line in lines:
        # Check for slide separator (## Slide X: pattern)
        if line.startswith('## Slide ') or (line.strip() == '---' and current_slide):
            if current_slide:
                slides.append('\n'.join(current_slide))
                current_slide = []
            if line.startswith('## Slide '):
                current_slide.append(line)
        else:
            current_slide.append(line)
    
    # Add last slide
    if current_slide:
        slides.append('\n'.join(current_slide))
    
    return [s.strip() for s in slides if s.strip()]


def convert_markdown_to_html(md_content: str) -> str:
    """Convert markdown content to HTML."""
    html = md_content
    
    # Convert headers
    html = re.sub(r'^## Slide \d+: (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Convert code blocks
    html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    
    # Convert inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Convert bold and italic
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)
    
    # Convert blockquotes
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Convert tables
    html = convert_tables(html)
    
    # Convert bullet lists
    html = convert_lists(html)
    
    # Convert horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)
    
    # Wrap remaining paragraphs
    lines = html.split('\n')
    result = []
    for line in lines:
        if line.strip() and not line.strip().startswith('<'):
            result.append(f'<p>{line}</p>')
        else:
            result.append(line)
    
    return '\n'.join(result)


def convert_tables(html: str) -> str:
    """Convert markdown tables to HTML."""
    lines = html.split('\n')
    result = []
    in_table = False
    table_lines = []
    
    for line in lines:
        if '|' in line and not line.strip().startswith('```'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                result.append(render_table(table_lines))
                in_table = False
                table_lines = []
            result.append(line)
    
    if in_table:
        result.append(render_table(table_lines))
    
    return '\n'.join(result)


def render_table(lines: list) -> str:
    """Render a markdown table as HTML."""
    if len(lines) < 2:
        return '\n'.join(lines)
    
    html = '<table>\n<thead>\n<tr>\n'
    
    # Header row
    headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
    for h in headers:
        html += f'<th>{h}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'
    
    # Data rows (skip separator line)
    for line in lines[2:]:
        if line.strip() and '---' not in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            html += '<tr>\n'
            for cell in cells:
                html += f'<td>{cell}</td>\n'
            html += '</tr>\n'
    
    html += '</tbody>\n</table>'
    return html


def convert_lists(html: str) -> str:
    """Convert bullet/numbered lists to HTML."""
    lines = html.split('\n')
    result = []
    in_list = False
    list_type = None
    
    for line in lines:
        stripped = line.strip()
        
        # Check for bullet list
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list or list_type != 'ul':
                if in_list:
                    result.append(f'</{list_type}>')
                result.append('<ul>')
                in_list = True
                list_type = 'ul'
            content = stripped[2:]
            result.append(f'<li>{content}</li>')
        # Check for numbered list
        elif re.match(r'^\d+\. ', stripped):
            if not in_list or list_type != 'ol':
                if in_list:
                    result.append(f'</{list_type}>')
                result.append('<ol>')
                in_list = True
                list_type = 'ol'
            content = re.sub(r'^\d+\. ', '', stripped)
            result.append(f'<li>{content}</li>')
        else:
            if in_list:
                result.append(f'</{list_type}>')
                in_list = False
                list_type = None
            result.append(line)
    
    if in_list:
        result.append(f'</{list_type}>')
    
    return '\n'.join(result)


def generate_html_slides(input_path: Path, output_path: Path):
    """Generate HTML slides from markdown file."""
    print(f"[READ] Reading markdown from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Parse into slides
    slides = parse_markdown_to_slides(markdown_content)
    print(f"[INFO] Found {len(slides)} slides")
    
    # Convert each slide to HTML
    slides_html = []
    for i, slide_md in enumerate(slides, 1):
        slide_html = convert_markdown_to_html(slide_md)
        slide_class = 'slide title-slide' if i == 1 else 'slide'
        slides_html.append(f'<div class="{slide_class}">\n{slide_html}\n</div>')
    
    # Generate final HTML
    final_html = HTML_TEMPLATE.format(
        title="TPR Drop Analysis After New Scanner Introduction - Slides",
        slides_content='\n\n'.join(slides_html),
        **COLORS
    )
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"[OK] HTML slides generated: {output_path}")
    print(f"\n[INFO] To create PDF:")
    print(f"   1. Open {output_path} in your browser")
    print(f"   2. Press Ctrl+P (or Cmd+P on Mac)")
    print(f"   3. Select 'Save as PDF' as destination")
    print(f"   4. Click Save")


# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# NotebookLM-Style Slide Generator")
    print("#"*60 + "\n")
    
    generate_html_slides(INPUT_FILE, OUTPUT_HTML)
    
    print("\n" + "#"*60)
    print("# Generation Complete!")
    print("#"*60 + "\n")
