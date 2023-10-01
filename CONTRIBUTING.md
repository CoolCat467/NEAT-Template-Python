# Contributing to this project

Thank you so much for your interest in contributing! All types of contributions are encouraged and valued. We welcome contributions from anyone willing to work in good faith with other contributors and the community.

There are many ways to contribute, no contribution is too small, and all contributions are valued. For example, you could:

- Use NEAT-Template-Python in a project, and give us feedback on what worked and what
  didn't.
- Write a blog post about your experiences with NEAT-Template-Python, good or bad.
- Improve documentation.
- Comment on issues.
- Add tests.
- Fix bugs.
- Add features.

We want contributing to be enjoyable and mutually beneficial; this
document tries to give you some tips to help that happen.
If you have thoughts on how it can be improved, then please let us know.


Getting started
---------------

If you're new to open source in general, you might find it useful to
check out [opensource.guide's How to Contribute to Open Source tutorial](https://opensource.guide/how-to-contribute/), or if
video's more your thing, [egghead.io has a short free video course](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github>).

NEAT-Template-Python is developed on GitHub. Code
and documentation changes are made through pull requests (see
`Preparing Pull Requests` below).


Providing support
-----------------

When helping others use NEAT-Template-Python, please remember that you are
representing our community, and we want this to be a friendly and
welcoming place.

Please remember that the authors and users of competing projects are
smart, thoughtful people doing their best to balance complicated and
conflicting requirements, just like us. Of course it's totally fine to
make specific technical critiques ("In project X, this is handled by
doing Y, NEAT-Template-Python does Z instead, which I prefer because...") or talk
about your personal experience ("I tried using X but I got super
frustrated and confused"), but refrain from generic statements like "X
sucks" or "I can't believe anyone uses X".

Please try not to make assumptions about people's gender, and in
particular remember that we're not all dudes. If you don't have a
specific reason to assume otherwise, then [singular they](https://en.wikipedia.org/wiki/Third-person_pronoun#Singular_they) makes a fine pronoun, and there are plenty of gender-neutral
collective terms: "Hey folks", "Hi all", ...

We also like the Recurse Center's [social rules](https://www.recurse.com/manual#sub-sec-social-rules):

* no feigning surprise (also available in a [sweet comic version](https://jvns.ca/blog/2017/04/27/no-feigning-surprise/))
* no well-actually's
* no subtle -isms ([more details](https://www.recurse.com/blog/38-subtle-isms-at-hacker-school))


Preparing pull requests
-----------------------

If you want to submit a documentation or code change, then that's done
by preparing a Github pull request (or "PR" for short).
We'll do our best to review your PR quickly. If it's
been a week or two and you're still waiting for a response, feel free
to post a comment poking us. (This can just be a comment with the
single word "ping"; it's not rude at all.)

Here's a quick checklist for putting together a good PR, with details
in separate sections below:

* `Pull Request Scope`: Does your PR address a single,
  self-contained issue?

* `Pull Request Tests`: Are your tests passing? Did you add any
  necessary tests? Code changes pretty much always require test
  changes, because if it's worth fixing the code then it's worth
  adding a test to make sure it stays fixed.

* `Pull Request Formatting`: If you changed Python code, then did
  you run ``black src tests``?

* `Pull Request Docs`: Did you make any necessary documentation
  updates?

* License: by submitting a PR, you're offering your
  changes under this project's license.



What to put in a PR
-----------------------

Each PR should, as much as possible, address just one issue and be
self-contained. If you have ten small, unrelated changes, then go
ahead and submit ten PRs – it's much easier to review ten small
changes than one big change with them all mixed together, and this way
if there's some problem with one of the changes it won't hold up all
the others.

If you're uncertain about whether a change is a good idea and want
some feedback before putting time into it, feel free to ask in an
issue or in the discussions tab.  If you have a partial change that you want
to get feedback on, feel free to submit it as a PR. (In this case it's
traditional to start the PR title with `[WIP]`, for "work in
progress".)

When you are submitting your PR, you can include ``Closes #123``,
``Fixes: #123`` or [some variation](https://help.github.com/en/articles/closing-issues-using-keywords) in either your commit message or the PR description, in order to
automatically close the referenced issue when the PR is merged.
This keeps us closer to the desired state where each open issue reflects some
work that still needs to be done.


Tests
-----

We use [pytest](https://pytest.org/) for testing. To run the tests
locally, you should run:

```shell
cd path/to/repo/checkout/
pip install -r ../checkout  # possibly using a virtualenv
pytest src
```

This doesn't try to be completely exhaustive – it only checks that
things work on your machine. But it's
a good way to quickly check that things seem to be working, and we'll
automatically run the full test suite when your PR is submitted, so
you'll have a chance to see and fix any remaining issues then.

You can use ``# pragma: no cover`` to mark lines where
lack-of-coverage isn't something that we'd want to fix (as opposed to
it being merely hard to fix). For example:

```python
else:  # pragma: no cover
    raise AssertionError("this can't happen!")
```

Some rules for writing good tests:

* Tests MUST pass deterministically

* Tests should never sleep unless *absolutely* necessary.

* We like tests to exercise real functionality.

* For cases where real testing isn't relevant or sufficient, then we
  strongly prefer fakes or stubs over mocks. Useful articles:

  * [Test Doubles - Fakes, Mocks and Stubs](https://dev.to/milipski/test-doubles---fakes-mocks-and-stubs)

  * [Mocks aren't stubs](https://martinfowler.com/articles/mocksArentStubs.html)

  * [Write test doubles you can trust using verified fakes](https://codewithoutrules.com/2016/07/31/verified-fakes/)

Writing reliable tests for obscure corner cases is often harder than
implementing a feature in the first place, but stick with it: it's
worth it! And don't be afraid to ask for help. Sometimes a fresh pair
of eyes can be helpful when trying to come up with devious tricks.


Code formatting
---------------

Instead of wasting time arguing about code formatting, we use [black](https://github.com/psf/black) as well as other tools to automatically
format all our code to a standard style. While you're editing code you
can be as sloppy as you like about whitespace; and then before you commit,
just run:

```shell
pip install -U pre-commit
pre-commit
```

to fix it up. (And don't worry if you forget – when you submit a pull
request then we'll automatically check and remind you.) Hopefully this
will let you focus on more important style issues like choosing good
names, writing useful comments, and making sure your docstrings are
nicely formatted. (black doesn't reformat comments or docstrings.)

If you would like, you can even have pre-commit run before you commit by
running:
```shell
pre-commit install
```

and now pre-commit will run before git commits. You can uninstall the
pre-commit hook at any time by running:
```shell
pre-commit uninstall
```

Very occasionally, you'll want to override black formatting. To do so,
you can can add ``# fmt: off`` and ``# fmt: on`` comments.

If you want to see what changes black will make, you can use:
```shell
black --diff src tests
```
(``--diff`` displays a diff, versus the default mode which fixes files
in-place.)


Additionally, in some cases it is necessary to disable isort changing the
order of imports. To do so you can add ``# isort: split`` comments.
For more information, please see [isort's docs](https://pycqa.github.io/isort/docs/configuration/action_comments.html).


Commit messages
---------------

We don't enforce any particular format on commit messages. In your
commit messages, try to give the context to explain *why* a change was
made.

The target audience for release notes is users, who want to find out
about changes that might affect how they use the library, or who are
trying to figure out why something changed after they upgraded.

The target audience for commit messages is some hapless developer
(think: you in six months... or five years) who is trying to figure
out why some code looks the way it does. Including links to issues and
any other discussion that led up to the commit is *strongly*
recommended.


Managing issues
---------------

As issues come in, they need to be responded to, tracked, and –
hopefully! – eventually closed.

As a general rule, each open issue should represent some kind of task
that we need to do. Sometimes that task might be "figure out what to
do here", or even "figure out whether we want to address this issue";
sometimes it will be "answer this person's question". But if there's
no followup to be done, then the issue should be closed.
