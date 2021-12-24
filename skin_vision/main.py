import typer


app = typer.Typer()


@app.command()
def train():
    pass


@app.command()
def predict():
    pass


def main(name: str):
    typer.echo(f"Hello {name}")


if __name__ == "__main__":
    typer.run(main)
