# Mode Zion
### A graphical AI technology with llama2, Stable Diffusion, and a BarkAI Voice

Development branch includes BarkAI Text to speech. The User inputs a prompt and ai replies with 3 models. First the image model will generate an image, then the language model generates a text reply, then Bark TTS reads the message aloud.

### Screenshots:
![Screenshot from 2023-07-30 00-15-33](https://github.com/graylan0/ModeZion/assets/34530588/5fa93ebe-d4ac-4a60-b36f-cb8cade99450)


![Screenshot from 2023-07-30 00-40-04](https://github.com/graylan0/ModeZion/assets/34530588/9eafe437-8005-4b81-a4a8-9a038d9d689a)


### Installation
``` Installation For Windows 10/11 with Nvidia-GPU```

1. Install Automatic1111  `sd.webui.zip` from `v1.0.0-pre here` https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre and extract the zip file.

2. Continue the Automatic1111 installation by running `update.bat`.
   
4. After running update.bat, Make `webui.bat` look like this. by adding `set COMMANDLINE_ARGS=--api`

![image](https://github.com/graylan0/ModeZion/assets/34530588/3d0c4be9-61ca-4936-9216-11b6916ee98a)

5. Start Automatic1111 by running `run.bat` . This should start the Automatic1111 WebUI and load the model to be used with the GUI program.
   
6. Download each part of the model/gui 7zip from here https://github.com/graylan0/ModeZion/releases/tag/v1 then extract the GUI/Model Folder with 7zip.

7. Open the Model/GUI folder that was extracted.

8. Run `llama-stable-gui.exe`. 
