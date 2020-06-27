## Running Jekyll in Docker

The initial step is to build your Docker image with Jekyll environment:

```sh
docker build --no-cache -f dockerfile/Dockerfile -t jekyll_env .
```

Then run it with port and directory mapping options:

 ```sh
docker run --rm -ti -p 8080:4000 -v $PWD:/src/ jekyll_env  bash
```

To serve your site just type following command in docker container bash command line:

```sh
jekyll serve
```

To serve your site with drafts:

```sh
jekyll serve --drafts
```

Serve the site and auto-regenerating on changes

```sh
jekyll serve --livereload
```

To access your site use following links: "http://localhost:8080" or "http://127.0.0.1:8080"

If you're using Docker Toolbox on Windows, try accessing "http://192.168.99.100:8080" instead.

## Convert Jupyter Notebook to Markdown

1. download the following GitHub gists jekyll.py and jekyll.tpl;
2. copy jekyll.py into root project directory and copy jekyll.tpl to any directory;
3. change pathes in jekyll.py file;
4. then type the following command into terminal:

```sh
jupyter nbconvert --to markdown <notebook_filename>.ipynb --config ../jekyll.py
```
