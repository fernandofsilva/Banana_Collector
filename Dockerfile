FROM python:3.6-slim

LABEL maintainer="Fernando Silva <fernando.f.silva@outlook.com>"

# Copy all files to app folder
COPY .. /app

# Set working directory
WORKDIR /app

# Install packages
RUN apt-get clean \
        && apt-get update \
        && apt-get install -y \
        git \
        unzip \
        && rm -rf /var/apt/lists/*

## Upgrade pip
RUN pip3 install --upgrade pip

# Install Udacity environment
RUN git clone https://github.com/udacity/deep-reinforcement-learning.git
RUN pip3 install deep-reinforcement-learning/python/.

# Extract Banana Collector environment
#RUN unzip unity/Banana_Linux_NoVis.zip -d deep-reinforcement-learning/p1_navigation
RUN unzip unity/Banana_Linux_NoVis.zip -d .

# Start jupyter notebook (uncomment to change the environment and run the
# container through the command:
#   docker run -p 8888:8888 fernandofsilva/banana_collector
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# Run python script
ENTRYPOINT ["python", "/app/codes/main.py"]