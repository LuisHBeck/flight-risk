install_requirements:
	@pip install --upgrade pip
	@pip install -r requirements.txt

load_env:
	@echo "Configuring environment variables..."
	@if [ ! -f .env ]; then cp .env.sample .env; fi
	direnv allow .

dev_setup: install_requirements load_env

fetch_raw_dataset:
	@echo "Fetching raw dataset from ANAC's repo..."
	mkdir -p .data
	@python utils/fetch_data.py

fetch_consolidated_dataset:
	@echo "Fetching consolidate dataset from project's repo..."
	mkdir -p .data
	@gdown $(CONSOLIDATED_FILE_ID) --output .data
