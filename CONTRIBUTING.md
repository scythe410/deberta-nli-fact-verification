# Contribution workflow

1. Update your local copy of `main` or `dev`.
2. Create a feature branch, for example `feature/word-overlap`.
3. Make and verify your changes, then push the feature branch.
4. Open a pull request from the feature branch into `dev`.
5. Verify the combined changes in `dev`.
6. Open a pull request from this repository's `dev` branch into `main`.

Both `main` and `dev` require pull requests. Direct pushes, force pushes,
and branch deletion are blocked, including for administrators.
Pull request discussions must be resolved before merging.

The required `Branch flow` check only allows this repository's `dev` branch
as the source of pull requests into `main`. It checks branch metadata;
it does not run application tests or establish model accuracy.

Reviewer approvals are currently optional so the repository owner can merge
while working alone. Review changes before merging and add required approvals
when additional collaborators are available.
