
# using python3.9:slim base image 
FROM python:3.11-slim

# sets the working directory in the CONTAINER
WORKDIR /api  

# install SYSTEM LEVEL packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# copy requirements.txt from . (current dir where dockerfile is) 
# into container's working directory (i.e /api)
COPY requirements.txt .

# install packages listed in requirements.txt
RUN pip install -r requirements.txt 

# where left '.' means current directory (where dockerfile is)
# right '.' means current workdir in container (as set by WORKDIR)
# copy everything from CURRENT DIR to WORKDIR
COPY . . 

# exposing port 5000 of the CONTAINER
EXPOSE 5000   

CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8080"]





