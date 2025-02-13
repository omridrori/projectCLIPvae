import torch
import clip
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import os
from typing import Dict, List, Union
import torch.nn.functional as F
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import dotenv

load_dotenv()


class TemplateGenerator:
    def __init__(self, api_key: str):
        """Initialize the template generator with OpenAI API key"""
        print(f"\nInitializing Template Generator...")
        print(f"API Key provided: {'Yes' if api_key else 'No'}")

        try:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-0125",
                temperature=0.7,
                openai_api_key=api_key
            )
            print("Successfully initialized ChatOpenAI")
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {str(e)}")
            raise

        # Create a clean, single-line prompt template
        template = (
            "Generate natural language templates that describe a person's appearance for CLIP image classification.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Each template should be a simple, clear sentence focusing ONLY on the specific attribute value\n"
            "- DO NOT add any assumptions or extra information beyond the attribute value\n"
            "- DO NOT specify age ranges, specific details, or additional characteristics\n"
            "- DO NOT use subjective or interpretive descriptions\n"
            "- FOCUS on distinguishing between the categories within each attribute\n"
            "- Keep descriptions general and factual\n"
            "- MAKE SURE to use the exact attribute values provided, without modifications\n\n"
            "For example, if given:\n"
            "ethnicity: African, Asian, European\n\n"
            "Expected Output:\n"
            '{{\n'
            '    "ethnicity": {{\n'
            '        "African": [\n'
            '            "a photograph of a person with African features",\n'
            '            "an African person",\n'
            '            "a face with African characteristics",\n'
            '            "a portrait of an African person",\n'
            '            "a person of African descent"\n'
            '        ],\n'
            '        "Asian": [\n'
            '            "a photograph of a person with Asian features",\n'
            '            "an Asian person",\n'
            '            "a face with Asian characteristics",\n'
            '            "a portrait of an Asian person",\n'
            '            "a person of Asian descent"\n'
            '        ],\n'
            '        "European": [\n'
            '            "a photograph of a person with European features",\n'
            '            "a European person",\n'
            '            "a face with European characteristics",\n'
            '            "a portrait of a European person",\n'
            '            "a person of European descent"\n'
            '        ]\n'
            '    }}\n'
            '}}\n\n'
            "Now, please generate templates for the following attributes and values:\n"
            "{attributes}\n\n"
            "The output should be in the same JSON format as the example.\n"
            "Remember to keep descriptions simple and focused only on the specific attribute value.\n"
            "Only include the JSON output, no additional text."
        )

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["attributes"]
        )

    def generate_templates(self, attribute_values: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """Generate templates using GPT for given attributes and values"""
        try:
            # Format attributes for the prompt
            attributes_str = "\n".join(
                f"{attr}: {', '.join(values)}"
                for attr, values in attribute_values.items()
            )

            print("\nGenerating templates for attributes:")
            print(attributes_str)

            # Generate the prompt using the template
            prompt = self.prompt.format(attributes=attributes_str)

            print("\nSending request to GPT...")
            # Get response from GPT
            response = self.llm.predict(prompt)

            print("\nReceived response from GPT. First 200 characters:")
            print(response[:200] + "..." if len(response) > 200 else response)

            # Parse the JSON response
            try:
                templates = json.loads(response)
                print("\nSuccessfully parsed JSON response")
                # Debug: Print template structure
                print("\nTemplate structure:")
                for attr, values in templates.items():
                    print(f"\n{attr}:")
                    for val, temps in values.items():
                        print(f"  {val}: {len(temps)} templates")
                return templates
            except json.JSONDecodeError as e:
                print(f"\nError parsing GPT response as JSON: {str(e)}")
                print("Raw response:", response)
                return None

        except Exception as e:
            print(f"\nError in template generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class CustomAttributeClassifier:
    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 openai_api_key: str = None,
                 debug: bool = False):
        self.device = device
        self.debug = debug
        print(f"Initializing CLIP model on {device}")

        # First try to get the API key from the parameter, then from environment
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if self.api_key:
            print("OpenAI API key found!")
            self.template_generator = TemplateGenerator(self.api_key)
        else:
            print("No OpenAI API key found in parameters or environment variables.")
            self.template_generator = None

        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.templates = None
        self.attribute_mappings = {}

    def normalize_templates(self, gpt_templates: Dict) -> Dict:
        """
        Normalize templates from GPT format to our indexed format
        """
        normalized = {}
        for attr, values in gpt_templates.items():
            normalized[attr] = {}
            # Create a mapping from value to index
            value_to_idx = {value: idx for idx, value in self.attribute_mappings[attr].items()}

            # Convert string keys to numeric indices
            for value, templates in values.items():
                if value in value_to_idx:
                    idx = value_to_idx[value]
                    normalized[attr][idx] = templates
                else:
                    print(f"Warning: Value '{value}' not found in mappings for {attr}")

        return normalized

    def print_templates(self):
        """Print the full template structure in a clear, organized format"""
        print("\nCurrent Templates:")
        print("=" * 50)

        for attr, templates in self.templates.items():
            print(f"\n{attr.upper()}:")
            print("-" * 30)

            # Sort by index to maintain consistent order
            for idx in sorted(templates.keys()):
                value = self.attribute_mappings[attr][idx]
                print(f"\n{value}:")
                for i, template in enumerate(templates[idx], 1):
                    print(f"  {i}. {template}")

        print("\n" + "=" * 50 + "\n")

    def create_text_templates(self, attribute_values: Dict[str, List[str]]) -> Dict:
        """Create text templates using GPT if available, otherwise use default templates"""
        # First create attribute mappings
        self.attribute_mappings = {
            attr: {idx: value for idx, value in enumerate(values)}
            for attr, values in attribute_values.items()
        }

        if self.template_generator:
            print("\nAttempting to generate templates using GPT...")
            try:
                gpt_templates = self.template_generator.generate_templates(attribute_values)
                if gpt_templates:
                    print("Successfully generated templates using GPT!")
                    # Normalize the templates to use indices
                    self.templates = self.normalize_templates(gpt_templates)
                    return self.templates

            except Exception as e:
                print(f"\nError during GPT template generation: {str(e)}")
                print("Falling back to custom templates.")
        else:
            print("\nGPT template generator not available (no API key provided).")
            print("Using custom templates instead.")

        # Fallback to default templates
        templates = {}
        for attr, values in attribute_values.items():
            templates[attr] = {}
            for idx, value in enumerate(values):
                templates[attr][idx] = [
                    f"a photo of a person with {value} appearance",
                    f"a {value} person",
                    f"a face with {value} features",
                    f"this person appears to be {value}",
                    f"a portrait showing {value} characteristics"
                ]

        self.templates = templates
        return templates

    def classify_images(self,
                       img_dir: str,
                       attribute_values: Dict[str, List[str]],
                       batch_size: int = 8) -> pd.DataFrame:
        """Classify images according to specified attributes using CLIP"""
        if not os.path.exists(img_dir):
            raise ValueError(f"Directory not found: {img_dir}")

        # Create or get templates
        if not self.templates:
            print("Generating templates...")
            self.create_text_templates(attribute_values)
            # Print templates once at initialization
            self.print_templates()

        print(f"\nScanning directory: {img_dir}")
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not image_files:
            raise ValueError(f"No image files found in {img_dir}")

        print(f"Found {len(image_files)} images")

        results = {'image_id': []}
        for attr in attribute_values.keys():
            results[attr] = []

        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size),
                     desc="Classifying images",
                     position=0,
                     leave=True):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            valid_files = []

            # Load and preprocess images
            for img_file in batch_files:
                try:
                    img_path = os.path.join(img_dir, img_file)
                    image = Image.open(img_path).convert('RGB')
                    processed_image = self.preprocess(image)
                    batch_images.append(processed_image)
                    valid_files.append(img_file)
                except Exception as e:
                    if self.debug:
                        print(f"Error processing {img_file}: {str(e)}")
                    continue

            if not batch_images:
                continue

            try:
                image_batch = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image_batch)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    for attr, values in attribute_values.items():
                        try:
                            if self.debug:
                                print(f"\nProcessing attribute: {attr}")
                                print(f"Values: {values}")

                            all_text_features = []
                            templates = self.templates[attr]

                            for idx, value in enumerate(values):
                                try:
                                    # Try both integer and string keys
                                    if idx in templates:
                                        key = idx
                                    elif str(idx) in templates:
                                        key = str(idx)
                                    else:
                                        raise KeyError(f"Neither key {idx} nor '{idx}' found in templates")

                                    texts = templates[key]
                                    text_tokens = clip.tokenize(texts).to(self.device)
                                    text_features = self.model.encode_text(text_tokens)
                                    text_features = text_features.mean(dim=0, keepdim=True)
                                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                    all_text_features.append(text_features)

                                except Exception as e:
                                    if self.debug:
                                        print(f"Error processing value {value}:")
                                        print(f"Full error: {str(e)}")
                                        import traceback
                                        traceback.print_exc()
                                    raise

                            text_features = torch.cat(all_text_features)
                            similarity = (100.0 * image_features @ text_features.T)
                            predictions = similarity.softmax(dim=-1).cpu().numpy()
                            class_indices = predictions.argmax(axis=1)

                            results[attr].extend(class_indices.tolist())

                        except Exception as e:
                            if self.debug:
                                print(f"Error processing attribute {attr}:")
                                print(f"Full error: {str(e)}")
                                import traceback
                                traceback.print_exc()
                            raise

                results['image_id'].extend(valid_files)

            except Exception as e:
                if self.debug:
                    print(f"Error processing batch:")
                    print(f"Full error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

        df = pd.DataFrame(results)
        self._print_distribution_statistics(df)
        return df

    def _print_distribution_statistics(self, df: pd.DataFrame):
        """Print distribution statistics for each attribute"""
        print(f"\nProcessed {len(df)} images successfully")
        print("\nAttribute distribution:")

        for attr, mapping in self.attribute_mappings.items():
            print(f"\n{attr.capitalize()} Distribution:")
            print("-" * 30)

            counts = df[attr].value_counts().sort_index()
            total = len(df)

            for idx in sorted(mapping.keys()):
                count = counts.get(idx, 0)
                percentage = (count / total) * 100
                print(f"{mapping[idx]:<25}: {count:>6,} ({percentage:>6.1f}%)")


def prepare_attributes_for_vae(df: pd.DataFrame,
                               attribute_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """Prepare the classified attributes for use in the VAE"""
    vae_df = df.copy()

    for attr, values in attribute_mapping.items():
        vae_df[attr] = vae_df[attr].astype(int)

    return vae_df