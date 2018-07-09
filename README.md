# PUMPS 2018

## Setup

Clone this repository to get the project folder.

    git clone https://github.com/illinois-impact/ece408_project.git

Download the rai binary for your platform.
You will probably use it for development, and definitely use it for submission.


| Operating System | Architecture | Stable Version (0.2.23) Link                                                             | Beta Version (0.2.31) Link                                                              |
| ---------------- | ------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Linux            | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.31/linux-amd64.tar.gz)   | [URL](https://github.com/rai-project/rai/releases/download/latest/linux-amd64.tar.gz)   |
| OSX/Darwin       | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.31/darwin-amd64.tar.gz)  | [URL](https://github.com/rai-project/rai/releases/download/latest/darwin-amd64.tar.gz)  |
| Windows          | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.31/windows-amd64.tar.gz) | [URL](https://github.com/rai-project/rai/releases/download/latest/windows-amd64.tar.gz) |

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS) or `%HOME%/.rai_profile` (Windows).
Your `.rai_profile` should look something like this (indented with tabs!)

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: <your-affiliation>

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).

## Labs

- [Device Query](labs/device_query)
- [Scatter](labs/scatter)
- [Gather](labs/gather)
- [Binning](labs/binning)


## Offline Development


## Issues


Please use the [Github issue manager] to report any issues or suggestions.

Include the outputs of

```bash
rai version
```

as well as the output of

```bash
rai buildtime
```

In your bug report. You can also invoke the `rai` command with verbose and debug outputs using

```bash
rai --verbose --debug
```

[github issue manager]: https://github.com/illinois-impact/pumps/issues

## License

NCSA/UIUC © [Abdul Dakkak](http://impact.crhc.illinois.edu/Content_Page.aspx?student_pg=Default-dakkak)

