# STNWeb v1.1 on local environment

Steps to run STNWeb in local environment using docker.

1. To execute the following command, navigate to the repository's root folder.

    ```bash ./docker/create-environment.sh```

    Docker will generate an image containing all the required STNWeb.

2. Next, execute the following command to make the Rest API and web accessible.

    ```bash ./docker/open-environment.sh```

3. Finally, you can now access the web on port 8081.

    [http://localhost:8081](http://localhost:8081)

    To solve discrete or continuous problems, utilize the sample files found in the [/test](/test/) folder. For instance, in the pmed6 case, there are separate files for each algorithm's execution such as `aco_out.txt` and `brkga_out`.txt. These files are available for your use.

-----

Cite:
```
@article{CHACONSARTORI2023100558,
    title = {STNWeb: A new visualization tool for analyzing optimization algorithms},
    journal = {Software Impacts},
    volume = {17},
    pages = {100558},
    year = {2023},
    issn = {2665-9638},
    doi = {https://doi.org/10.1016/j.simpa.2023.100558},
    url = {https://www.sciencedirect.com/science/article/pii/S2665963823000957},
    author = {Camilo {Chacón Sartori} and Christian Blum and Gabriela Ochoa},
    keywords = {Algorithm analysis, Visualization, Behavior of optimization algorithms, Web application},
    abstract = {STNWeb is a new web tool for the visualization of the behavior of optimization algorithms such as metaheuristics. It allows for the graphical analysis of multiple runs of multiple algorithms on the same problem instance and, in this way, it facilitates the understanding of algorithm behavior. It may help, for example, in identifying the reasons for a rather low algorithm performance. This, in turn, can help the algorithm designer to change the algorithm in order to improve its performance. STNWeb is designed to be user-friendly. Moreover, it is offered for free to the research community.}
}
```

For more details, see the [PDF](/tutorial/tutorial_stnweb_v1_1.pdf) in the folder [/tutorial](/tutorial/).
