# Adding examples to the gallery

You have a cool application of HuracanPy? We would love for you to share it with the world to showcase what can be done with the package!

The recommend format for example pages in Jupyter Notebooks. While it is preferable that examples are based on HuracanPy's embedded example data, on downloadable data, or on data that is generated within the example, it is possible to use your own data. In that case, the notebook cannot be updated when the documentation is re-built (see NB below).

1. Create a notebook with your code in `docs/examples/`, some explanations, and your potential nice images.
2. By default the thumbnail image in the gallery will be the last image. To use a different image, add `nbsphinx-thumbnail` to the tags of the cell you want to use. See [here](https://nbsphinx.readthedocs.io/en/latest/gallery/cell-tag.html) for more details.
3. (Optional) Check out the result by building the doc locally. Run `make html` in the `docs` folder and open the html files in `docs/_build/html/`
4. Commit and push the changes to your own fork.
5. Create a pull request to add your example to the online documentation. 

NB: Notebooks may or may not be re-run at each documentation build. While re-running helps with keeping the documentation up-to-date, we cannot re-run your notebook if it uses data that is on your own computer. We also do not wish to re-run computations that may take significant time every time.

* For your notebook to be re-run, you need to clear cell's output: In the Edit Menu, select "Clear Outputs of All Cells" before saving your notebook. 
* For your notebook *not* to be re-run, do the opposite: Be careful to run all cells before saving the notebook. 