## Prerequisites

- **Docker** installed on your system im using buildx for information!
- **X11 server** for GUI display forwarding

# EXtra information:

- do not forget the add the group to the user
```bash
sudo usermod -aG docker $USER
```
for admin permissions

- and add start the xhost for gui display + also u can add your .profile

```bash
xhost +local:docker
```
or you can add by this in terminal:

```bash
echo 'xhost +local:docker > /dev/null 2>&1' >> ~/.bashrc
```

- and if u just installed docker and try to run it for sudo purposes u need to add the docker the system such as ```newgrp docker```
or restart computer!

- **If u just restart computer everything will be better** 

## Build the Image

Run the following command in this directory (`handin_mehmetbahadursun`)

```bash
docker buildx build -t finalproject_mehmetbahadursun .
```

## Running the Container

**Docker run command:**

```bash
docker run -it --device=/dev/dri \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v $(pwd):/home/robotics/workspace \
     finalproject_mehmetbahadursun
```
## Robotic Part

Most of the information's are gaven inside of the code.
For running the simulation execute:

```
robotics/main.py
```

In here u have a option of running integration task
or robotic part 3 task. 

## Vision part

For see the outputs open the ```Debug``` mode
inside of run_all.py
