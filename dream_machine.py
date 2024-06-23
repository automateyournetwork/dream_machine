import os
import json
import yaml
import torch
import random
import requests
import torchaudio
import streamlit as st
from datetime import datetime
from diffusers import StableDiffusion3Pipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from einops import rearrange
from moviepy.editor import VideoFileClip, AudioFileClip
from huggingface_hub import login

# Ensure required keys are initialized
if 'hf_key' not in st.session_state:
    st.session_state['hf_key'] = None
if 'image_prompt' not in st.session_state:
    st.session_state['image_prompt'] = ""
if 'music_prompt' not in st.session_state:
    st.session_state['music_prompt'] = ""
if 'final_video' not in st.session_state:
    st.session_state['final_video'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 1

# Define the function to send requests to the models
def send_request(model, prompt):
    url = f"http://ollama:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "prompt": f"Generate only the prompt with the following details: {prompt}. Do not include anything additional only the prompt.",
        "stream": False,
        "keep_alive": 0
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json().get('response', '')
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Load selections from JSON or YAML file
def load_selections(file_path):
    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
        with open(file_path, 'r') as f:
            selections = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or YAML.")
    return selections

# Set up the Streamlit app
st.set_page_config(page_title="AI Animation Generator", layout="wide")

# Page 1: Input Hugging Face API key and download models
def page1():
    st.title("Dream Machine - Bring Your Dreams to Life")
    st.title("Step 1: Enter Hugging Face API Key")
    hf_key = st.text_input("Hugging Face API Key", type="password")
    if st.button("Next"):
        if hf_key:
            with st.spinner("Logging in and downloading models... this may take a while"):
                login(token=hf_key)
            st.session_state['page'] = 2
            st.experimental_rerun()
        else:
            st.error("Please enter a valid Hugging Face API Key.")

# Page 2: Input image prompt
def page2():
    st.title("Dream Machine - Bring Your Dreams to Life")
    st.title("Step 2: Enter Image Prompt")

    prompt_option = st.radio(
        "Choose how to generate the image prompt:",
        ('Enter manually', 'Use dropdown menus')
    )

    selections = load_selections('image_selections.yaml')  # Load from YAML file
    for key in selections:
        selections[key].insert(0, "")
        selections[key].insert(1, "RANDOM")

    if prompt_option == 'Enter manually':
        image_prompt = st.text_area("Image Prompt", height=200)
    else:
        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            setting = st.selectbox("Choose a setting or type your own:", selections['settings'] + ["Other (Type your own)"], key="setting")
            if setting == "RANDOM":
                setting = random.choice(selections['settings'][2:])  # Exclude '' and 'RANDOM' itself

            custom_setting = ""
            if setting == "Other (Type your own)":
                custom_setting = st.text_input("Type your own setting:", key="custom_setting")
                setting = custom_setting

            character = st.selectbox("Choose a character type or type your own:", selections['characters'] + ["Other (Type your own)"], key="character")
            if character == "RANDOM":
                character = random.choice(selections['characters'][2:])  # Exclude '' and 'RANDOM' itself
            if character == "Other (Type your own)":
                character = st.text_input("Type your own character:", key="custom_character")

            activity = st.selectbox("Choose an activity or type your own:", selections['activities'] + ["Other (Type your own)"], key="activity")
            if activity == "RANDOM":
                activity = random.choice(selections['activities'][2:])  # Exclude '' and 'RANDOM' itself
            if activity == "Other (Type your own)":
                activity = st.text_input("Type your own activity:", key="custom_activity")

            environment = st.selectbox("Choose an environment or type your own:", selections['environments'] + ["Other (Type your own)"], key="environment")
            if environment == "RANDOM":
                environment = random.choice(selections['environments'][2:])  # Exclude '' and 'RANDOM' itself
            if environment == "Other (Type your own)":
                environment = st.text_input("Type your own environment:", key="custom_environment")

            artistic_style = st.selectbox("Choose an artistic style or type your own:", selections['artistic_styles'] + ["Other (Type your own)"], key="artistic_style")
            if artistic_style == "RANDOM":
                artistic_style = random.choice(selections['artistic_styles'][2:])  # Exclude '' and 'RANDOM' itself
            if artistic_style == "Other (Type your own)":
                artistic_style = st.text_input("Type your own artistic style:", key="custom_artistic_style")

        with col2:
            movie = st.selectbox("Choose a movie or type your own:", selections['movies'] + ["Other (Type your own)"], key="movie")
            if movie == "RANDOM":
                movie = random.choice(selections['movies'][2:])  # Exclude '' and 'RANDOM' itself
            if movie == "Other (Type your own)":
                movie = st.text_input("Type your own movie:", key="custom_movie")

            celebrity = st.selectbox("Choose a celebrity or type your own:", selections['celebrities'] + ["Other (Type your own)"], key="celebrity")
            if celebrity == "RANDOM":
                celebrity = random.choice(selections['celebrities'][2:])  # Exclude '' and 'RANDOM' itself
            if celebrity == "Other (Type your own)":
                celebrity = st.text_input("Type your own celebrity:", key="custom_celebrity")

            animal = st.selectbox("Choose an animal or type your own:", selections['animals'] + ["Other (Type your own)"], key="animal")
            if animal == "RANDOM":
                animal = random.choice(selections['animals'][2:])  # Exclude '' and 'RANDOM' itself
            if animal == "Other (Type your own)":
                animal = st.text_input("Type your own animal:", key="custom_animal")

            historic_event = st.selectbox("Choose a historic event or type your own:", selections['historic_events'] + ["Other (Type your own)"], key="historic_event")
            if historic_event == "RANDOM":
                historic_event = random.choice(selections['historic_events'][2:])  # Exclude '' and 'RANDOM' itself
            if historic_event == "Other (Type your own)":
                historic_event = st.text_input("Type your own historic event:", key="custom_historic_event")

            style = st.selectbox("Choose a style or type your own:", selections['styles'] + ["Other (Type your own)"], key="style")
            if style == "RANDOM":
                style = random.choice(selections['styles'][2:])  # Exclude '' and 'RANDOM' itself
            if style == "Other (Type your own)":
                style = st.text_input("Type your own style:", key="custom_style")

        if st.button("Generate Image Prompt", key="generate_prompt"):
            # Combine selected keywords into a simple prompt
            keywords = []
            if character:
                keywords.append(character)
            if activity:
                keywords.append(activity)
            if environment:
                keywords.append(f"in a {environment} setting")
            if setting:
                keywords.append(f"that is {setting}")
            if artistic_style:
                keywords.append(f"in the style of {artistic_style}")
            if movie:
                keywords.append(f"from the movie {movie}")
            if celebrity:
                keywords.append(f"featuring {celebrity}")
            if animal:
                keywords.append(f"with a {animal}")
            if historic_event:
                keywords.append(f"during {historic_event}")
            if style:
                keywords.append(f"with a {style} style")

            keywords_prompt = "using the following keywords can you create a prompt that will generate a single image: " + ", ".join(keywords) if keywords else "a general scene"
            image_prompt = send_request("llama3", keywords_prompt)
            st.session_state['generated_image_prompt'] = image_prompt
        else:
            image_prompt = st.session_state.get('generated_image_prompt', '')

        st.text_area("Generated Image Prompt", image_prompt, height=200)

    if st.button("Next", key="next_page2"):
        if image_prompt:
            st.session_state['image_prompt'] = image_prompt
            st.session_state['page'] = 3
            st.experimental_rerun()
        else:
            st.error("Please enter a valid image prompt.")

# Page 3: Input music prompt and generate animation
def page3():
    st.title("Dream Machine - Bring Your Dreams to Life")
    st.title("Step 3: Enter Music Prompt")

    prompt_option = st.radio(
        "Choose how to generate the music prompt:",
        ('Enter manually', 'Use dropdown menus')
    )

    selections = load_selections('audio_selections.yaml')  # Load from YAML file
    for key in selections:
        selections[key].insert(0, "")
        selections[key].insert(1, "RANDOM")

    if prompt_option == 'Enter manually':
        music_prompt = st.text_area("Music Prompt", height=200)
    else:
        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            mood = st.selectbox("Choose a mood or type your own:", selections['moods'] + ["Other (Type your own)"], key="mood")
            if mood == "RANDOM":
                mood = random.choice(selections['moods'][2:])  # Exclude '' and 'RANDOM' itself
            if mood == "Other (Type your own)":
                mood = st.text_input("Type your own mood:", key="custom_mood")

            genre = st.selectbox("Choose a genre or type your own:", selections['genres'] + ["Other (Type your own)"], key="genre")
            if genre == "RANDOM":
                genre = random.choice(selections['genres'][2:])  # Exclude '' and 'RANDOM' itself
            if genre == "Other (Type your own)":
                genre = st.text_input("Type your own genre:", key="custom_genre")

            instrument = st.selectbox("Choose an instrument or type your own:", selections['instruments'] + ["Other (Type your own)"], key="instrument")
            if instrument == "RANDOM":
                instrument = random.choice(selections['instruments'][2:])  # Exclude '' and 'RANDOM' itself
            if instrument == "Other (Type your own)":
                instrument = st.text_input("Type your own instrument:", key="custom_instrument")

            style = st.selectbox("Choose a style or type your own:", selections['styles'] + ["Other (Type your own)"], key="style")
            if style == "RANDOM":
                style = random.choice(selections['styles'][2:])  # Exclude '' and 'RANDOM' itself
            if style == "Other (Type your own)":
                style = st.text_input("Type your own style:", key="custom_style")

        with col2:
            composer = st.selectbox("Choose a composer or type your own:", selections['composers'] + ["Other (Type your own)"], key="composer")
            if composer == "RANDOM":
                composer = random.choice(selections['composers'][2:])  # Exclude '' and 'RANDOM' itself
            if composer == "Other (Type your own)":
                composer = st.text_input("Type your own composer:", key="custom_composer")

            decade = st.selectbox("Choose a decade or type your own:", selections['decades'] + ["Other (Type your own)"], key="decade")
            if decade == "RANDOM":
                decade = random.choice(selections['decades'][2:])  # Exclude '' and 'RANDOM' itself
            if decade == "Other (Type your own)":
                decade = st.text_input("Type your own decade:", key="custom_decade")

            setting = st.selectbox("Choose a setting or type your own:", selections['settings'] + ["Other (Type your own)"], key="setting")
            if setting == "RANDOM":
                setting = random.choice(selections['settings'][2:])  # Exclude '' and 'RANDOM' itself
            if setting == "Other (Type your own)":
                setting = st.text_input("Type your own setting:", key="custom_setting")

        if st.button("Generate Music Prompt", key="generate_music_prompt"):
            # Combine selected keywords into a simple prompt
            keywords = []
            if mood:
                keywords.append(mood)
            if genre:
                keywords.append(genre)
            if instrument:
                keywords.append(instrument)
            if style:
                keywords.append(f"in a {style} style")
            if composer:
                keywords.append(f"by {composer}")
            if decade:
                keywords.append(f"from the {decade}")
            if setting:
                keywords.append(f"set in a {setting}")

            keywords_prompt = "using the following keywords can you create a prompt that will generate a music track: " + ", ".join(keywords) if keywords else "a general track"
            music_prompt = send_request("llama3", keywords_prompt)
            st.session_state['generated_music_prompt'] = music_prompt
        else:
            music_prompt = st.session_state.get('generated_music_prompt', '')

        st.text_area("Generated Music Prompt", music_prompt, height=200)

    if st.button("Next", key="next_page3"):
        if music_prompt:
            st.session_state['music_prompt'] = music_prompt
            generate_animation()
            st.session_state['page'] = 4
            st.experimental_rerun()
        else:
            st.error("Please enter a valid music prompt.")

def generate_animation():
        image_prompt = st.session_state['image_prompt']
        music_prompt = st.session_state['music_prompt']

        # Creating the pipeline, huggingface's wrapper for easy model loading and inference
        sd3m_pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16)

        # We're using a GPU here so setting the device to enable cuda support
        sd3m_pipe = sd3m_pipe.to("cuda")

        # Enabling CPU offloading
        # Model offloading will only move a model component onto the GPU when it needs to be executed, while keeping the remaining components on the CPU.
        sd3m_pipe.enable_model_cpu_offload()

        # Generate the image
        sd3m_image = sd3m_pipe(
            prompt=image_prompt,
            negative_prompt="", # The prompt or prompts not to guide the image generation
            num_inference_steps=50, # The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            height=576,
            width=1024,
            guidance_scale=7.0, # Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
        ).images[0]

        # Save the image
        sd3m_image.save("sd3m_image.png")

        # Creating the Stable Video Diffusion pipeline and loading the model
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16")

        # Enabling cuda support since we're using a GPU
        svd_pipe = svd_pipe.to("cuda")

        # Enabling CPU model offloading
        svd_pipe.enable_model_cpu_offload()

        # Setting a seed
        svd_generator = torch.manual_seed(42)

        # Generating the frames
        frames = svd_pipe(sd3m_image,
                          decode_chunk_size=8,
                          generator=svd_generator).frames[0]

        # Save video
        export_to_video(frames, "diffusion_video.mp4", fps=7)

        # Loading the model using stable-audio-tools and grabbing configuration specs
        sao_model, sao_model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = sao_model_config["sample_rate"]
        sample_size = sao_model_config["sample_size"]

        # Setting model to cuda support
        device = "cuda"
        sao_model = sao_model.to(device)

        # Setting 'conditioning' aka the prompt and time
        sao_conditioning = [{
            "prompt": music_prompt,
            "seconds_start": 0,
            "seconds_total": 7
        }]

        # Generate stereo audio
        sao_output = generate_diffusion_cond(
            sao_model,                       # The model used for generating the audio. This is typically a pre-trained diffusion model.
            steps=100,                       # The number of steps for the diffusion process. Higher steps generally improve quality but increase computation time.
            cfg_scale=7,                     # The classifier-free guidance scale. A higher value can enhance the details but may also introduce artifacts.
            conditioning=sao_conditioning,   # The conditioning variable that guides the generation process, defined above.
            sample_size=sample_size,         # The size of the generated audio sample. This defines the length and resolution of the output.
            sigma_min=0.3,                   # The minimum value of the noise scale. Lower values can help in fine details.
            sigma_max=500,                   # The maximum value of the noise scale. Higher values can help in capturing the overall structure.
            sampler_type="dpmpp-3m-sde",     # The type of sampler used. Different samplers can affect the speed and quality of the generation.
            device=device                    # The device on which the computation is performed (e.g., 'cpu' or 'cuda' for GPU).
        )


        # Reshape the tensor to combine batches into a single continuous stream per channel, converting to stereo audio format.
        # Effectively concatenates the audio samples from all batches into a single continuous stream per channel for stereo audio so an audio library can play it.
        output = rearrange(sao_output, "b d n -> d (b n)")

        # Post processing, peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save("sao_output.wav", output, sample_rate)

        # Using Moviepy to combine the video with the audio
        video = VideoFileClip("diffusion_video.mp4")
        audio = AudioFileClip("sao_output.wav").subclip(0, 4)
        final_video = video.set_audio(audio)
        final_video.write_videofile("final_video.mp4", codec="libx264", audio_codec="aac")
        st.session_state['final_video'] = "final_video.mp4"

            # Cleanup: Remove intermediate files
        os.remove("sd3m_image.png")
        os.remove("sao_output.wav")
        os.remove("diffusion_video.mp4")

# Page 4: Display final animation with music
def page4():
    st.title("Dream Machine - Bring Your Dreams to Life")
    if st.session_state.get('final_video'):
        st.video(st.session_state['final_video'])
        st.write(f"**Image Prompt:** {st.session_state['image_prompt']}")
        st.write(f"**Music Prompt:** {st.session_state['music_prompt']}")
        st.download_button(
            label="Download Video",
            data=open(st.session_state['final_video'], "rb").read(),
            file_name="final_video.mp4",
            mime="video/mp4"
        )
        if st.button("Generate Another Animation", key="start_over"):
            os.remove("final_video.mp4")
            st.session_state['page'] = 2
    else:
        st.write("No video generated yet. Please go back and generate the video.")

# Navigation logic
if 'page' not in st.session_state:
    st.session_state['page'] = 1

if st.session_state['page'] == 1:
    page1()
elif st.session_state['page'] == 2:
    page2()
elif st.session_state['page'] == 3:
    page3()
elif st.session_state['page'] == 4:
    page4()
