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

Most of the information's are gaven inside of the code most of them named
as their work. and can be followable from gitlogs

Also in main.py its the architecture can be seenable.

If u want to tune or change something use config.yaml all of the params
and the settings initialized there.
All of the p2p points and relative things attached there the main code mostly
does not have hard-coded params and settings. So u can just modify config.yaml

If u run the code there will be option which you want to run such as
part3(box,cylinder,tbox) and part5(integration)

and in part3 u will be able to select rrt or p2p

For running the simulation execute:

```
robotics/main.py
```

In here u have a option of running integration task
or robotic part 3 task.

### Video Demonstrations

| Demo | Link |
|------|------|
| P2P Data Collection | [YouTube]([https://youtu.be/ljMGeYzblWg](https://youtu.be/JD8mP_SH8sg)) |
| P2P Execution | [YouTube](https://youtu.be/nj_jDb7eGxQ) |
| RRT Execution | [YouTube](https://youtu.be/tpHh2QvFxpo) |
| Integration (Part 5) | [YouTube](https://youtu.be/QDvPMWtlsyQ) | 

## Vision part

For see the outputs open the ```Debug``` mode
inside of run_all.py
