### Environment
The Environment I solved was the unity navigation problem where agent is tasked to collect yellow bananas while avoiding blue bananas. The Environment consist of 37 states with velocity and perception of object. Getting blue bananas result in -1 reward, while yellow bananas gives +1. To solve the environment. You need a score of 13+ for 100 consecutive episodes.

### Requirements Installation
To run the repo you need to install packages and download the Unity environment related to your operating system. We will start by create private python environment using Anaconda(other alternatives would also work. e.g. venv)
1. Make a private environment in python 3.7:
   * ```bash
     conda create -n dagster python=3.7
     ```
2. Install required libraries:
    * ```bash
      pip install -r requirements.txt
      ```

## Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, unzip and place the folder in this directory.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)