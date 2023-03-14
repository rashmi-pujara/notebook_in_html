This repository demonstrates:
1. A conversion of a Jupyter notebook into a static HTML
2. Serving this HTML using Github Pages

## Setup

1. `python -m venv ./venv/github-pages/` - Create a virtual environment
2. `.\venv\github-pages\Scripts\Activate.ps1` - Activate it (use the right script depending on your terminal, this one is for `Windows-PowerShell`)
3. `pip install -r requirements.txt` - install the packages into the virtual environment
4. `python -m ipykernel install --user --name=github_pages --display-name=github_pages` - set up the IPython kernel in the virtual environment, such that this environment could be used in VSCode. Notice the name that we provide to the environment. This is how it will be identified in VSCode. 

## Convert the Notebook to an HTML file

1. `jupyter nbconvert --to html ./notebooks/example.ipynb --output-dir ./docs --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_input_tags="['noshow']"` - converts the notebook into an HTML file. 

## Serve the HTML through Github Pages
(reference: [official docs](https://docs.github.com/en/pages/quickstart). Jekyll built-in support: [official docs](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll))
1. Enable Github Pages through the repository settings. For this the repository will have to be public, or your user should be on the Pro program. 
2. The public URL is: `GITHUB-USER-NAME.github.io/REPOSITORY-NAME`, and the landing page is `index.html`. NOte that you can you set your own domain if you own one, and this can be set via the repository's settings. 


## The scope of a project. 
The scope of a project for this class should involve training or fine-tuning a model for a ML task, using one of the models learned in class (such as RNN, LSTM, or Transformer). You are encouraged to focus on an interesting aspect of the problem, such as comparing different hyperparameters, data preprocessing, or model architectures. The specific problem and model you choose are up to you. The methodology should follow the framework reviewed in class, and the work should be original. You should aim to complete the project within one week of work. You may use datasets from class, or datasets available through Tensorflow or PyTorch, Kaggle, or other resources. In order to earn a perfect score, you are not expected to devise a new model architecture or optimizer, but rather to explore and fine-tune an existing model with hyperparameter tuning, and to be able to explain and present your exploration well in a written form.

