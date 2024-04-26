# Use the official Python 3.10 image as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /home/stud/abedinz1/localDisk/RAG/RAG

# Copy the Python script from the host to the container
COPY new_main.py .

# Run the Python script when the container starts
CMD ["python", "new_main.py"]

