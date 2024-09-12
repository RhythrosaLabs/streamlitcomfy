import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import replicate
import base64
import re
import asyncio
import aiohttp

# Constants
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
API_KEY_FILE = "api_keys.json"

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'openai': None, 'replicate': None}

if 'customization' not in st.session_state:
    st.session_state.customization = {
        'image_types': ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI'],
        'script_types': ['Player', 'Enemy', 'Game Object', 'Level Background'],
        'image_count': {t: 0 for t in ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI']},
        'script_count': {t: 0 for t in ['Player', 'Enemy', 'Game Object', 'Level Background']},
        'use_replicate': {'generate_music': False},
        'code_types': {'unity': False, 'unreal': False, 'blender': False},
        'generate_elements': {
            'game_concept': True,
            'world_concept': True,
            'character_concepts': True,
            'plot': True,
            'storyline': False,
            'dialogue': False,
            'game_mechanics': False,
            'level_design': False
        },
        'image_model': 'dall-e-3',
        'chat_model': 'gpt-4',
        'code_model': 'gpt-4',
    }

# Load API keys from a file
def load_api_keys():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            data = json.load(file)
            return data.get('openai'), data.get('replicate')
    return None, None

# Save API keys to a file
def save_api_keys(openai_key, replicate_key):
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"openai": openai_key, "replicate": replicate_key}, file)

# Get headers for OpenAI API
def get_openai_headers():
    return {
        "Authorization": f"Bearer {st.session_state.api_keys['openai']}",
        "Content-Type": "application/json"
    }

# Generate content using selected chat model
async def generate_content(prompt, role):
    if st.session_state.customization['chat_model'] in ['gpt-4', 'gpt-3.5-turbo']:
        data = {
            "model": st.session_state.customization['chat_model'],
            "messages": [
                {"role": "system", "content": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_API_URL, headers=get_openai_headers(), json=data) as response:
                    response_data = await response.json()
                    if "choices" not in response_data:
                        error_message = response_data.get("error", {}).get("message", "Unknown error")
                        return f"Error: {error_message}"

                    content_text = response_data["choices"][0]["message"]["content"]
                    return content_text

        except Exception as e:
            return f"Error: Unable to communicate with the OpenAI API: {str(e)}"
    elif st.session_state.customization['chat_model'] == 'llama':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "meta/llama-2-70b-chat",
                input={
                    "prompt": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development.\n\n{prompt}\n\n",
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "max_length": 500,
                    "repetition_penalty": 1
                }
            )
            return ''.join(output)
        except Exception as e:
            return f"Error: Unable to generate content using Llama: {str(e)}"
    else:
        return "Error: Invalid chat model selected."

# Generate images using selected image model
async def generate_image(prompt, size, steps=25, guidance=3.0, interval=2.0):
    if st.session_state.customization['image_model'] == 'dall-e-3':
        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": f"{size[0]}x{size[1]}",
            "n": 1,
            "response_format": "url"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(DALLE_API_URL, headers=get_openai_headers(), json=data) as response:
                    response_data = await response.json()
                    if "data" not in response_data:
                        error_message = response_data.get("error", {}).get("message", "Unknown error")
                        return f"Error: {error_message}"
                    if not response_data["data"]:
                        return "Error: No data returned from API."
                    return response_data["data"][0]["url"]
        except Exception as e:
            return f"Error: Unable to generate image: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SD Flux-1':
        try:
            width, height = size
            aspect_ratio = f"{width}:{height}"

            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])

            output = client.run(
                "black-forest-labs/flux-pro",
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "guidance": guidance,
                    "interval": interval,
                    "safety_tolerance": 2,
                    "output_format": "png",
                    "output_quality": 100
                }
            )
            return output
        except Exception as e:
            return f"Error: Unable to generate image using SD Flux-1: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SDXL Lightning':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "bytedance/sdxl-lightning-4step",
                input={"prompt": prompt}
            )
            return output[0] if output else None
        except Exception as e:
            return f"Error: Unable to generate image using SDXL Lightning: {str(e)}"
    else:
        return "Error: Invalid image model selected."

# Generate music using Replicate's MusicGen
async def generate_music(prompt):
    try:
        client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        output = client.run(
            "meta/musicgen",
            input={
                "prompt": prompt,
                "model_version": "stereo-large",
                "output_format": "mp3",
                "normalization_strategy": "peak"
            }
        )
        if isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return None
    except Exception as e:
        st.error(f"Error: Unable to generate music: {str(e)}")
        return None

# Generate multiple images based on customization settings
async def generate_images(customization, game_concept):
    images = {}

    image_prompts = {
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game...",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game...",
        'Background': "Create a wide, highly detailed background image for a level of the game...",
        'Object': "Create a detailed object image for a 2D game...",
        'Texture': "Generate a seamless texture pattern...",
        'Sprite': "Create a game sprite sheet with multiple animation frames...",
        'UI': "Design a cohesive set of user interface elements for a 2D game..."
    }

    sizes = {
        'Character': (1024, 1024),
        'Enemy': (1024, 1024),
        'Background': (1920, 1080),
        'Object': (512, 512),
        'Texture': (512, 512),
        'Sprite': (1024, 1024),
        'UI': (1024, 1024)
    }

    tasks = []
    for img_type in customization['image_types']:
        for i in range(customization['image_count'].get(img_type, 0)):
            prompt = f"{image_prompts[img_type]} The design should fit the following game concept: {game_concept}. Variation {i + 1}"
            size = sizes[img_type]
            task = asyncio.create_task(generate_image(prompt, size))
            tasks.append((task, f"{img_type.lower()}_image_{i + 1}"))

    for task, img_name in tasks:
        image_url = await task
        images[img_name] = image_url

    return images

# Generate scripts based on customization settings and code types
async def generate_scripts(customization, game_concept):
    script_descriptions = {
        'Player': "Create a comprehensive player character script for a 2D game. Include movement, input handling, and basic interactions.",
        'Enemy': "Develop a detailed enemy AI script for a 2D game. Include patrolling, player detection, and attack behaviors.",
        'Game Object': "Script a versatile game object that can be interacted with, collected, or activated by the player.",
        'Level Background': "Create a script to manage the level background in a 2D game, including parallax scrolling if applicable."
    }

    scripts = {}
    selected_code_types = customization['code_types']
    code_model = customization['code_model']

    tasks = []
    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            for code_type, selected in selected_code_types.items():
                if selected:
                    if code_type == 'unity':
                        lang = 'csharp'
                        file_ext = '.cs'
                    elif code_type == 'unreal':
                        lang = 'cpp'
                        file_ext = '.cpp'
                    elif code_type == 'blender':
                        lang = 'python'
                        file_ext = '.py'
                    else:
                        continue  # Skip if it's an unknown code type

                    desc = f"{script_descriptions[script_type]} The script should be for {code_type.capitalize()}. Generate ONLY the code, without any explanations or comments outside the code. Ensure the code is complete and can be directly used in a project."

                    task = asyncio.create_task(generate_content(desc, "game development"))
                    tasks.append((task, f"{script_type.lower()}_{code_type}_script_{i + 1}{file_ext}"))

    for task, script_name in tasks:
        script_code = await task

        # Clean up the generated code
        script_code = script_code.strip()
        script_code = re.sub(r'^```.*\n', '', script_code)  # Remove starting code block markers
        script_code = re.sub(r'\n```$', '', script_code)  # Remove ending code block markers

        scripts[script_name] = script_code

    return scripts

# Generate a complete game plan
async def generate_game_plan(user_prompt, customization):
    game_plan = {}

    # Status updates
    status = st.empty()
    progress_bar = st.progress(0)

    def update_status(message, progress):
        status.text(message)
        progress_bar.progress(progress)

    # Generate game elements
    elements_to_generate = customization['generate_elements']
    total_elements = sum(elements_to_generate.values())
    progress_increment = 0.6 / max(total_elements, 1)
    current_progress = 0.0

    for element, should_generate in elements_to_generate.items():
        if should_generate:
            update_status(f"Generating {element.replace('_', ' ')}...", current_progress)
            content = await generate_content(f"Create a detailed {element.replace('_', ' ')} for the following game concept: {user_prompt}", "game design")
            game_plan[element] = content
            current_progress += progress_increment

    # Generate images
    if any(customization['image_count'].values()):
        update_status("Generating game images...", 0.7)
        game_plan['images'] = await generate_images(customization, game_plan.get('game_concept', ''))

    # Generate scripts
    if any(customization['script_count'].values()):
        update_status("Writing game scripts...", 0.85)
        game_plan['scripts'] = await generate_scripts(customization, game_plan.get('game_concept', ''))

    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.95)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}"
        game_plan['music'] = await generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Function to display images
def display_image(image_url, caption):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad responses
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=caption, use_column_width=True)
    except requests.RequestException as e:
        st.warning(f"Unable to load image: {caption}")
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.warning(f"Unable to display image: {caption}")
        st.error(f"Error: {str(e)}")

# Streamlit app layout
st.set_page_config(page_title="Game Dev Automation", page_icon="🎮", layout="wide")
st.markdown('<style>' + open("style.css").read() + '</style>', unsafe_allow_html=True)

st.title("🎮 Game Dev Automation")

# Sidebar
with st.sidebar:
    st.markdown("## 🛠 Settings")

    # API Key Inputs with Tooltips
    with st.expander("🔑 API Keys"):
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_keys.get('openai', ''),
            type="password",
            help="Get your OpenAI API key from your account dashboard."
        )
        replicate_key = st.text_input(
            "Replicate API Key",
            value=st.session_state.api_keys.get('replicate', ''),
            type="password",
            help="Get your Replicate API key from your account settings."
        )
        if st.button("Save API Keys"):
            save_api_keys(openai_key, replicate_key)
            st.session_state.api_keys['openai'] = openai_key
            st.session_state.api_keys['replicate'] = replicate_key
            st.success("API Keys saved successfully!")

    # Model Selection with Help Text
    st.markdown("### 🤖 Model Selection")
    st.session_state.customization['chat_model'] = st.selectbox(
        "Select Chat Model",
        options=['gpt-4', 'gpt-3.5-turbo', 'llama'],
        index=0,
        help="Choose the language model for generating game concepts and narratives."
    )
    st.session_state.customization['image_model'] = st.selectbox(
        "Select Image Generation Model",
        options=['dall-e-3', 'SD Flux-1', 'SDXL Lightning'],
        index=0,
        help="Select the model for generating game images."
    )
    st.session_state.customization['code_model'] = st.selectbox(
        "Select Code Generation Model",
        options=['gpt-4', 'gpt-3.5-turbo', 'llama'],
        index=0,
        help="Choose the model for generating code scripts."
    )

    # Asset Library
    st.markdown("### 📂 Asset Library")
    if 'game_plan' in st.session_state and 'images' in st.session_state['game_plan']:
        for img_name, img_url in st.session_state['game_plan']['images'].items():
            if isinstance(img_url, str) and img_url.startswith('http'):
                st.image(img_url, caption=img_name, use_column_width=True)
                st.download_button("Download", img_url, file_name=f"{img_name}.png")

# Main content area with Tabs
options_tab, results_tab = st.tabs(["📝 Options", "📊 Results"])

with options_tab:
    st.markdown("## 🎮 Define Your Game")

    with st.form("generation_form"):
        user_prompt = st.text_area(
            "Describe your game idea:",
            "Enter a detailed description of your game here...",
            height=150,
            help="This description will be used as the foundation for generating all game assets."
        )

        st.markdown("### 🖼️ Image Generation")
        for img_type in st.session_state.customization['image_types']:
            st.session_state.customization['image_count'][img_type] = st.number_input(
                f"Number of {img_type} Images",
                min_value=0,
                value=st.session_state.customization['image_count'][img_type],
                key=f"image_count_{img_type}"
            )

        st.markdown("### 💻 Script Generation")
        for script_type in st.session_state.customization['script_types']:
            st.session_state.customization['script_count'][script_type] = st.number_input(
                f"Number of {script_type} Scripts",
                min_value=0,
                value=st.session_state.customization['script_count'][script_type],
                key=f"script_count_{script_type}"
            )

        st.markdown("### ⚙️ Code Type Selection")
        st.session_state.customization['code_types']['unity'] = st.checkbox(
            "Unity C# Scripts",
            value=st.session_state.customization['code_types']['unity'],
            key="unity"
        )
        st.session_state.customization['code_types']['unreal'] = st.checkbox(
            "Unreal C++ Scripts",
            value=st.session_state.customization['code_types']['unreal'],
            key="unreal"
        )
        st.session_state.customization['code_types']['blender'] = st.checkbox(
            "Blender Python Scripts",
            value=st.session_state.customization['code_types']['blender'],
            key="blender"
        )

        st.markdown("### 🔧 Additional Elements")
        st.session_state.customization['generate_elements']['storyline'] = st.checkbox(
            "Detailed Storyline",
            value=st.session_state.customization['generate_elements']['storyline']
        )
        st.session_state.customization['generate_elements']['dialogue'] = st.checkbox(
            "Sample Dialogue",
            value=st.session_state.customization['generate_elements']['dialogue']
        )
        st.session_state.customization['generate_elements']['game_mechanics'] = st.checkbox(
            "Game Mechanics Description",
            value=st.session_state.customization['generate_elements']['game_mechanics']
        )
        st.session_state.customization['generate_elements']['level_design'] = st.checkbox(
            "Level Design Document",
            value=st.session_state.customization['generate_elements']['level_design']
        )
        st.session_state.customization['use_replicate']['generate_music'] = st.checkbox(
            "Generate Background Music",
            value=st.session_state.customization['use_replicate']['generate_music']
        )

        generate_button = st.form_submit_button("Generate Game Plan")

if generate_button:
    if not st.session_state.api_keys['openai'] or not st.session_state.api_keys['replicate']:
        st.error("Please enter and save both OpenAI and Replicate API keys.")
    else:
        with st.spinner('Generating game plan...'):
            game_plan = asyncio.run(generate_game_plan(user_prompt, st.session_state.customization))
            st.session_state['game_plan'] = game_plan
        st.success('Game plan generated successfully!')

with results_tab:
    if 'game_plan' in st.session_state:
        st.markdown("## 📊 Generated Game Plan")

        if 'game_concept' in st.session_state['game_plan']:
            with st.expander("📖 Game Concept"):
                st.write(st.session_state['game_plan']['game_concept'])

        if 'world_concept' in st.session_state['game_plan']:
            with st.expander("🌍 World Concept"):
                st.write(st.session_state['game_plan']['world_concept'])

        if 'character_concepts' in st.session_state['game_plan']:
            with st.expander("🦸 Character Concepts"):
                st.write(st.session_state['game_plan']['character_concepts'])

        if 'plot' in st.session_state['game_plan']:
            with st.expander("🎭 Plot"):
                st.write(st.session_state['game_plan']['plot'])

        if 'images' in st.session_state['game_plan']:
            st.markdown("### 🖼️ Generated Images")
            for img_name, img_url in st.session_state['game_plan']['images'].items():
                if isinstance(img_url, str) and img_url.startswith('http'):
                    display_image(img_url, img_name)
                else:
                    st.write(f"{img_name}: {img_url}")

        if 'scripts' in st.session_state['game_plan']:
            st.markdown("### 💻 Generated Scripts")
            for script_name, script_code in st.session_state['game_plan']['scripts'].items():
                language = script_name.split('.')[-1]
                with st.expander(f"View {script_name}"):
                    st.code(script_code, language=language)

        if 'additional_elements' in st.session_state['game_plan']:
            st.markdown("### 🔧 Additional Elements")
            for element_name, element_content in st.session_state['game_plan']['additional_elements'].items():
                with st.expander(f"View {element_name.capitalize()}"):
                    st.write(element_content)

        # Save results
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add text documents
            for key in ['game_concept', 'world_concept', 'character_concepts', 'plot']:
                if key in st.session_state['game_plan']:
                    zip_file.writestr(f"{key}.txt", st.session_state['game_plan'][key])

            # Add images
            if 'images' in st.session_state['game_plan']:
                for asset_name, asset_url in st.session_state['game_plan']['images'].items():
                    if isinstance(asset_url, str) and asset_url.startswith('http'):
                        img_response = requests.get(asset_url)
                        img = Image.open(BytesIO(img_response.content))
                        img_file_name = f"{asset_name}.png"
                        with BytesIO() as img_buffer:
                            img.save(img_buffer, format='PNG')
                            zip_file.writestr(img_file_name, img_buffer.getvalue())

            # Add scripts
            if 'scripts' in st.session_state['game_plan']:
                for script_name, script_code in st.session_state['game_plan']['scripts'].items():
                    zip_file.writestr(script_name, script_code)

            # Add additional elements
            if 'additional_elements' in st.session_state['game_plan']:
                for element_name, element_content in st.session_state['game_plan']['additional_elements'].items():
                    zip_file.writestr(f"{element_name}.txt", element_content)

            # Add music if generated
            if 'music' in st.session_state['game_plan'] and st.session_state['game_plan']['music']:
                try:
                    music_response = requests.get(st.session_state['game_plan']['music'])
                    music_response.raise_for_status()
                    zip_file.writestr("background_music.mp3", music_response.content)
                except requests.RequestException as e:
                    st.error(f"Error downloading music: {str(e)}")

        st.download_button(
            "Download Game Plan ZIP",
            zip_buffer.getvalue(),
            file_name="game_plan.zip",
            mime="application/zip",
            help="Download a ZIP file containing all generated assets and documents."
        )

        # Display generated music if applicable
        if 'music' in st.session_state['game_plan'] and st.session_state['game_plan']['music']:
            st.markdown("### 🎵 Generated Music")
            st.audio(st.session_state['game_plan']['music'], format='audio/mp3')
        else:
            st.warning("No music was generated or an error occurred during music generation.")
    else:
        st.info("Generate a game plan to see the results here.")

# Footer
st.markdown("---")
st.markdown("""
    Created by [Your Name](https://your-website.com) | 
    [GitHub](https://github.com/your-github) | 
    [Twitter](https://twitter.com/your-twitter) | 
    [Instagram](https://instagram.com/your-instagram)
    """, unsafe_allow_html=True)

# Initialize Replicate client
if st.session_state.api_keys['replicate']:
    replicate.Client(api_token=st.session_state.api_keys['replicate'])

# Load API keys on startup
openai_key, replicate_key = load_api_keys()
if openai_key and replicate_key:
    st.session_state.api_keys['openai'] = openai_key
    st.session_state.api_keys['replicate'] = replicate_key
