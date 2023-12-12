# use python runtime
FROM python:3.8-slim

# Set the working directory in container
WORKDIR /app/

# Copy the Python script
COPY app.py /app/

COPY templates /app/templates

# Copy the requirements file
COPY requirements.txt /tmp/

# Copy the model file
COPY final_refined_prediction_model.pkl /app/


# Install the required python packages
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

#Ste the maintainer label
LABEL maintainer="Steven Daniel <danielsteven.ds@gmail.com>"

# upgrade pip
RUN pip install --upgrade pip


#make port 7860 available to the world outside tthis container
EXPOSE 5000

#Run app.pywhen the container launches
CMD ["python", "app.py"]

