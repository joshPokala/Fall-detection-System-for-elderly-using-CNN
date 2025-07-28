# Fall Detection CNN Model

Model to detect falls through Computer Vision

## Instructions to run

### Prerequisites:
> Python 3.11.4 or lower (tensorflow does not support newer Python versions)

### To install required Python packages:
while inside the project directory:

```pip install -r requirements.txt```

### Set up environment variables:

open `example.env` and modify the RECIPIENT_EMAIL to own email.
e.g.
```
/example.env
...
...
RECIPIENT_EMAIL=test@example.com
```

then rename the file to `.env`

### Running
The project can be run with our trained model by running
```python main.py```

This will start a Flask web server at http://localhost:5000 which will then display the realtime model. Web frontend is still a work in progress.