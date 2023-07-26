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

For more details, see the [PDF](/tutorial/tutorial_stnweb_v1_1.pdf) in the folder [/tutorial](/tutorial/).