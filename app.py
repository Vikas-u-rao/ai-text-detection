import gradio as gr
from transformers import pipeline
import torch
import re
import os
from docx import Document

class AITextDetector:
    def __init__(self):
        self.classifier = None
        self.load_model()
    
    def load_model(self):
        """Load the AI text detection model from Hugging Face"""
        try:
            print("Loading AI text detection model...")
            self.classifier = pipeline(
                "text-classification",
                model="VSAsteroid/ai-text-detector-hc3",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
    
    def detect_text(self, input_text):
        """
        Detect if text is AI-generated or human-written
        Returns: (label, confidence_score, confidence_bar_html)
        """
        if not input_text.strip():
            return "Please enter some text to analyze.", 0.0, ""
        
        if self.classifier is None:
            return "Model not loaded. Please try again.", 0.0, ""
        
        try:
            # Run inference
            results = self.classifier(input_text)
            
            # Extract results - model returns scores for both labels
            ai_score = 0.0
            human_score = 0.0
            
            for result in results[0]:
                if "AI" in result['label'].upper() or "GENERATED" in result['label'].upper():
                    ai_score = result['score']
                else:
                    human_score = result['score']
            
            # Determine the prediction
            if ai_score > human_score:
                label = "AI-Generated"
                confidence = ai_score
            else:
                label = "Human-Written"
                confidence = human_score
            
            # Create confidence visualization
            confidence_percentage = confidence * 100
            confidence_bar = self.create_confidence_bar(confidence_percentage, label)
            
            return label, f"{confidence_percentage:.2f}%", confidence_bar
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", 0.0, ""
    
    def create_confidence_bar(self, confidence_percentage, label):
        """Create an HTML confidence bar"""
        color = "#ff6b6b" if "AI" in label else "#51cf66"
        return f"""
        <div style="margin: 10px 0;">
            <div style="font-weight: bold; margin-bottom: 5px;">Confidence: {confidence_percentage:.2f}%</div>
            <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
                <div style="background-color: {color}; height: 100%; width: {confidence_percentage}%; 
                           border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    def create_text_confidence_bar(self, confidence_percentage, label):
        """Create a text-based confidence bar for markdown display"""
        # Create a text-based progress bar
        bar_length = 20
        filled_length = int(bar_length * confidence_percentage / 100)
        bar_char = "█" if "AI" in label else "▓"
        empty_char = "░"
        
        bar = bar_char * filled_length + empty_char * (bar_length - filled_length)
        emoji = "🤖" if "AI" in label else "👤"
        
        return f"{emoji} **Confidence:** {confidence_percentage:.1f}% `{bar}`"
    
    def extract_text_from_file(self, file_path):
        """Extract text content from uploaded files"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Remove markdown formatting for better analysis
                    content = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
                    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
                    content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic
                    content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code
                    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Remove links
                    return content
            
            elif file_extension == '.docx':
                doc = Document(file_path)
                text_content = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content.append(paragraph.text)
                return '\n'.join(text_content)
            
            else:
                return f"Unsupported file format: {file_extension}. Please upload .txt, .md, or .docx files."
                
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def analyze_file(self, file_obj):
        """Analyze uploaded file for AI text detection"""
        if file_obj is None:
            return "Please upload a file to analyze.", "", ""
        
        try:
            # Extract text from file
            text_content = self.extract_text_from_file(file_obj.name)
            
            if text_content.startswith("Error") or text_content.startswith("Unsupported"):
                return text_content, "", ""
            
            # Check if file is too large or too small
            if len(text_content.strip()) < 10:
                return "File content is too short for analysis (minimum 10 characters).", "", ""
            
            if len(text_content) > 50000:  # Limit to ~50k characters
                text_content = text_content[:50000]
                truncation_note = "\n\n*Note: File was truncated to 50,000 characters for analysis.*"
            else:
                truncation_note = ""
            
            # Split into chunks if text is very long
            if len(text_content) > 5000:
                return self.analyze_long_text(text_content, truncation_note)
            else:
                # Analyze the entire text
                label, confidence_str, conf_bar = self.detect_text(text_content)
                confidence_num = float(confidence_str.replace('%', ''))
                text_bar = self.create_text_confidence_bar(confidence_num, label)
                
                file_info = f"**File:** {os.path.basename(file_obj.name)}\n"
                file_info += f"**Length:** {len(text_content)} characters\n\n"
                
                result = f"{file_info}**Overall Result:** {label} ({confidence_str})\n\n{text_bar}{truncation_note}"
                
                return result, conf_bar, text_content[:500] + "..." if len(text_content) > 500 else text_content
                
        except Exception as e:
            return f"Error analyzing file: {str(e)}", "", ""
    
    def analyze_long_text(self, text_content, truncation_note=""):
        """Analyze long text by splitting into chunks"""
        # Split text into paragraphs or sentences
        chunks = self.split_text_into_chunks(text_content)
        
        results = []
        ai_count = 0
        human_count = 0
        total_confidence = 0
        
        results.append(f"**File Analysis Results** ({len(chunks)} sections analyzed)\n")
        results.append("=" * 50 + "\n")
        
        for i, chunk in enumerate(chunks, 1):
            if len(chunk.strip()) < 20:  # Skip very short chunks
                continue
                
            label, confidence_str, _ = self.detect_text(chunk)
            confidence_num = float(confidence_str.replace('%', ''))
            text_bar = self.create_text_confidence_bar(confidence_num, label)
            
            if "AI" in label:
                ai_count += 1
            else:
                human_count += 1
            
            total_confidence += confidence_num
            
            results.append(f"### Section {i}")
            results.append(f"*{chunk[:200]}{'...' if len(chunk) > 200 else ''}*\n")
            results.append(f"**Result:** {label} ({confidence_str})")
            results.append(text_bar)
            results.append("\n" + "-" * 30 + "\n")
        
        # Overall summary
        total_sections = ai_count + human_count
        if total_sections > 0:
            avg_confidence = total_confidence / total_sections
            overall_label = "Predominantly AI-Generated" if ai_count > human_count else "Predominantly Human-Written"
            
            results.insert(2, f"**Overall Assessment:** {overall_label}\n")
            results.insert(3, f"**AI Sections:** {ai_count} | **Human Sections:** {human_count}\n")
            results.insert(4, f"**Average Confidence:** {avg_confidence:.1f}%\n\n")
        
        results.append(truncation_note)
        
        return "\n".join(results), "", ""
    
    def split_text_into_chunks(self, text, max_chunk_size=1000):
        """Split long text into analyzable chunks"""
        # First try splitting by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If we still have chunks that are too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                sentences = re.split(r'[.!?]+\s+', chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk + sentence) <= max_chunk_size:
                        temp_chunk += sentence + ". "
                    else:
                        if temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + ". "
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
        
        return final_chunks

# Initialize the detector
detector = AITextDetector()

def analyze_single_text(text):
    """Wrapper function for single text analysis"""
    label, confidence, conf_bar = detector.detect_text(text)
    return label, confidence, conf_bar

def analyze_uploaded_file(file_obj):
    """Wrapper function for file analysis"""
    return detector.analyze_file(file_obj)

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="AI Text Detection Tool",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .description {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        <div class="main-header">
        #  AI Text Detection Tool
        </div>
        <div class="description">
        Detect whether text was written by Artificial Intelligence or Humans.
        </div>
        """)
        
        with gr.Tabs():
            # Single Text Analysis Tab
            with gr.TabItem("Single Text Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        single_input = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="Paste or type the text you want to analyze here...",
                            lines=8,
                            max_lines=15
                        )
                        single_button = gr.Button("Analyze Text", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        single_label = gr.Textbox(label="Prediction", interactive=False)
                        single_confidence = gr.Textbox(label="Confidence", interactive=False)
                        single_conf_bar = gr.HTML(label="Confidence Visualization")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["Artificial intelligence is a rapidly evolving field that encompasses machine learning, natural language processing, and computer vision. These technologies are transforming industries and creating new possibilities for automation and innovation."],
                        ["I woke up this morning feeling refreshed after a good night's sleep. The sun was shining through my bedroom window, and I could hear birds chirping outside. It reminded me of my childhood summers at my grandmother's house."],
                        ["The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes."]
                    ],
                    inputs=single_input,
                    label="Try these examples:"
                )
            
            # File Upload Analysis Tab
            with gr.TabItem("File Upload Analysis"):
                gr.Markdown("### Upload and Analyze Files")
                gr.Markdown("Upload text files (.txt), Markdown files (.md), or Word documents (.docx) for AI text detection analysis.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload File",
                            file_types=[".txt", ".md", ".docx"],
                            type="filepath"
                        )
                        file_button = gr.Button("Analyze File", variant="primary", size="lg")
                        
                        gr.Markdown("**Supported formats:**")
                        gr.Markdown("- 📄 `.txt` - Plain text files")
                        gr.Markdown("- 📝 `.md` - Markdown files") 
                        gr.Markdown("- 📋 `.docx` - Word documents")
                        
                    with gr.Column(scale=2):
                        file_results = gr.Markdown(
                            label="Analysis Results", 
                            value="Upload a file and click 'Analyze File' to see results here..."
                        )
                
                with gr.Row():
                    with gr.Column():
                        file_confidence_bar = gr.HTML(label="Confidence Visualization")
                    with gr.Column():
                        file_preview = gr.Textbox(
                            label="File Preview (first 500 characters)",
                            lines=8,
                            interactive=False
                        )
        
        # Event handlers
        single_button.click(
            fn=analyze_single_text,
            inputs=single_input,
            outputs=[single_label, single_confidence, single_conf_bar]
        )
        
        file_button.click(
            fn=analyze_uploaded_file,
            inputs=file_input,
            outputs=[file_results, file_confidence_bar, file_preview]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Model:** VSAsteroid/ai-text-detector-hc3 from Hugging Face  
        **Note:** This tool provides predictions based on the model's training data. Results should be used as guidance, not definitive proof.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    print("Starting AI Text Detection Web App...")
    interface = create_interface()
    
    # Launch with public sharing option for deployment
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True
    )