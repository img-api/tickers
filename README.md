# Recipe

Gen-AI App for proposing recipes based on ingredients


## Requirements:

You need to have the following installed

- Python 3.11+ (should work with previous versions)
- poetry


### Install dependencies

- Run 'poetry install' in the root folder
- You should have postgres running using docker-compose:
    docker-compose up -d


## Execution

Using command-line:

- Run 'poetry shell' in the root folder.
- Navigate to the folder 'src/ticker_extractor'
- customize the .env file indicating your local IP
- Load tickers in the database with command `python3 tickers.py`
- Run 'fastapi dev main.py' or 'fastapi run main.py' for starting the web server


## API Examples

It's possible to call the api endpoint using localhost:8000 as base url:

### For text:

- https://localhost:8000/recommend_recipe?ingredients=chicken,garlic

### For images:

You can use any image number from 1 to 10. The system will autocomplete the image path:

- https://localhost:8000/recommend_recipe?image=2

## Architecture

- See `docs/architecture_diagram.png`

## TODO
- Sanitize input.
- Add authentication (optional).
- Use a queue to avoid API limits overriding.

## Environment used:
- WSL2 (requires using local network IP)
- Ubuntu 24.02


