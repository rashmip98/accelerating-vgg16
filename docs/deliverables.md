# Deliverables

Your final report, along with the code submission, presentation and reported FOM, will help us grade your final project.

## Deadlines

* Presentation in class on 12/08
* Final report due on 12/13 at 11:59 PM
* Source code on GitHub Classroom is due by 12/13 11:59 PM

## Presentation

The slide deck should contain the following slides:

* Title slide
* Baseline Design (Software)
* Baseline Design (Hardware)
* Design Space Exploration (Optimizations)
* Preliminary Results

The *baseline design* should relate back to the non-optimized version specified in the milestones. The *design space exploration and optimizations* refers to your optimization flow and the specific optimizations you carried out. The *preliminary results* should contain a CPU baseline and results from your current, preliminary implementation.

### Presentation Order

TBD

### Template

You can find the presentation template on Canvas in the *Final Project* module.

### Rubric

Your presentation grade will be decided by rubric with the following equally weighted categories:

* Poise
* Length of presentation
* Organization
* Visuals
* Data Presentation
* Mechanics

See the rubric posted on the Canvas assignment for more details into these categories.

## Report

Your final report should contain no more than 10 pages, excluding references. The report should have the following sections:

* Abstract
* Introduction
* Methodology
* Evaluation
* Conclusion
* Appendix (no more than 4 pages)

You must use the [ACM Master Article Template](https://www.acm.org/publications/proceedings-template). We suggest using [Overleaf](https://www.overleaf.com/).

## Code

Please go to the link below to create a Github team. The team should be the same name as the team name on Canvas.

The link is TBD.

If you have a pre-existing GitHub repository, we suggest to push your changes to the new remote created by the link above and then to continue using the new remote instead of the old remote. Before following this, both students should make sure their repository is up to date with the old remote (pull push). Then, student A should run the following commands

```bash
git pull
git remote set-url origin <new_github_url>
git push
```

Once student A has completed running those commands, student B should clone the new repository into a new directory.
1. `git clone <new_github_url>`

In order to validate your results, we will run the following commands with `test` being replaced with your group project github url:

```bash
git clone https://github.com/penn-ese-539-2021S/project-group-test
cd project-group-test
aws_ppi
run_script benchmark.ppi # this is ran in aws_ppi
```

**Make sure that you can succeed at running the listed four commands. We will be unable to verify your FOM if the command fails.**

`benchmark.ppi` should be an aws_ppi command script that runs your benchmark script and reports the runtimes, accuracies and FOM of your design. `benchmark.ppi` should be stored in the root directory of your repository. An example aws_ppi command script is shown below. More details can be found [here](https://cmd2.readthedocs.io/en/latest/features/scripting.html#command-scripts).


```bash
launch fpga_build
start_job ...
```
