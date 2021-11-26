# Acquiring Lab 4 Code

Submitting your final project codebase requires you to sign up for the final project on GitHub Classroom then to clone from the repository created for you.

To sign up for the final project, visit this link: [https://classroom.github.com/a/caqKZBl6](https://classroom.github.com/a/caqKZBl6).

The link will prompt you for a group name. Please use the group number on Canvas as your group name.

Visit your newly created repository then clone your repository using the HTTPS option. An example command for a user with the group name test would have a git clone command similar to this:

```bash
git clone https://github.com/penn-ese-539-2021S/project-group-0
```

Once cloned, follow the rest of these handouts making sure to commit your changes often. Ideally, you should commit your changes at least at the completion of each milestone.

If you have a pre-existing GitHub repository, we suggest pushing your changes to the new remote created by the link above and then continuing using the new remote instead of the old remote. Before following this, both students should make sure their repository is up to date with the old remote (pull push). Then, student A should run the following commands

```bash
git pull
git remote set-url origin <new_github_url>
git push
```

Once student A has completed running those commands, student B should clone the new repository into a new directory

```bash
git clone <new_github_url>
```