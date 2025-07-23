install.packages('devtools', repos = "http://cran.us.r-project.org")

library(devtools)
# install.packages(c('usethis', 'pkgdown', 'rcmdcheck', 'roxygen2', 'rversions', 'urlchecker'))
install.packages("mvtnorm", repos="http://cran.us.r-project.org")

# install GSAR
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos="http://cran.us.r-project.org")
BiocManager::install(version = "3.17", ask = FALSE)
BiocManager::install("GSAR")

# install WassersteinGoF
install_github("gmordant/WassersteinGoF", ref = "main")