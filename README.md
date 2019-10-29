# ML Foundation - SP-Recommendation-Algorithm

This is our student project for the Machine Learning Foundation course done by Pr. Fragkiskos Malliaros at Centrale Supélec Paris, October - December 2019.

This is an attempt to create a recommendation algorithm that includes two user-set parameters: specificity and popularity of recommendation.

Specificity means how close to the user profile the suggestion will be (based on similar users' profiles).

Popularity is a threshold to avoid too popular titles and look for less seen series (based on several measures of popularity).

Our database is from MyAnimeList.net, a platform where ~300.000 users publicly rate and comment ~14.000 animated series called "Animes". Source data found on kaggle: https://www.kaggle.com/azathoth42/myanimelist

## Installing the project

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

What things you need to have installed

```
Python 3.0+
	Numpy
	Pickle
```

### Installing

First, you need to clone the project as is from GitHub.

Then you need to download the source data and the treated data (~1.7GB zipped, ~10GB unzipped).

Google Drive link to data archive: https://drive.google.com/open?id=1qt_-5IzO4MU41iV79bjCWu_d7X17H699

Please extract the content of this archive directly inside the project's folder. It is ignored by git.

## Authors

* **Zixuan Feng**
* **Thomas Lamson**
* **Jinshuo Wu**
* **Yiming Wu**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
