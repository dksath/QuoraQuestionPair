# Running Prototype

## Model
1. clone this repo.
    ```
    git clone https://github.com/dksath/QuoraQuestionPair
    ```

2. Download the model weights from `(enter gdrive link)` 

3. Unzip file and add model folder into the repo


## FastAPI
To run the model in the prototype, install `FastAPI` and `uvicorn` server. More information at [FastAPI](https://fastapi.tiangolo.com/) official website.

1. Install dependenices:
       - `FastAPI`
       - `Uvicorn`
       - `TensorFlow`
       - `pandas`
       - `transformers` (HuggingFace)
       - `numpy`
    
    Install dependenices by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
    ```powershell
    conda install -c conda-forge fastapi uvicorn

    ```    
    Or, Install dependenices by pip:
    
    ```powershell
    pip install fastapi "uvicorn[standard]"

    ```
    Or install relevant dependencies from requirements.txt:
    ```powershell
    pip install -r requirements.txt
    ```

2. run the backend server :
    ```powershell
    uvicorn main:app 
    ```

## React
1. Install `node.js` and `npm`.

    **You’ll need to have Node 14.0.0 or later version on your local development machine** (but it’s not required on the server). We recommend using the latest LTS version. You can use [nvm](https://github.com/creationix/nvm#installation) (macOS/Linux) or [nvm-windows](https://github.com/coreybutler/nvm-windows#node-version-manager-nvm-for-windows) to switch Node versions between different projects. For more information visit the official website of [node.js](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

2. Install dependency `axios`
   
    Install dependenices by [npm](https://docs.npmjs.com/cli/v8/commands/npm-install):
    
    ```powershell
    npm install [<package-spec> ...]

    aliases: add, i, in, ins, inst, insta, instal, isnt, isnta, isntal, isntall
    ``` 
3. change directory into `react` folder and start the server

    ```powershell
    npm start
    ``` 
