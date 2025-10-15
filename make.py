import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console

# Inicializa o Typer e o Rich para uma saída bonita
app = typer.Typer(
    rich_markup_mode="markdown",
    help="""
    **Python Project Command-Line Interface**

    Este script substitui o Makefile tradicional para gerenciar o projeto.
    """,
)
console = Console()

# Constantes do projeto
PYTHON_VERSION = "3.12"
ROOT_DIR = Path(__file__).parent.resolve()

# --- Funções de Comando ---

def _run_command(command: List[str], exit_on_error: bool = True):
    """Executa um comando no shell e lida com erros."""
    try:
        process = subprocess.run(
            command, 
            check=True, 
            text=True, 
            capture_output=True,
            # Garante que os comandos 'uv' sejam executados no ambiente virtual correto
            env=os.environ
        )
        if process.stdout:
            console.print(process.stdout)
        if process.stderr:
            console.print(f"[yellow]Stderr:[/yellow]\n{process.stderr}")
        return process.returncode
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Erro ao executar o comando: {' '.join(command)}[/bold red]")
        console.print(f"[red]Exit Code:[/red] {e.returncode}")
        console.print(f"[red]Stdout:[/red]\n{e.stdout}")
        console.print(f"[red]Stderr:[/red]\n{e.stderr}")
        if exit_on_error:
            sys.exit(e.returncode)
    except FileNotFoundError:
        console.print(f"[bold red]Comando não encontrado: {command[0]}. Certifique-se de que está instalado e no PATH.[/bold red]")
        if exit_on_error:
            sys.exit(1)


@app.command()
def requirements():
    """Instala as dependências Python com 'uv sync'."""
    console.print("[bold green]Instalando dependências...[/bold green]")
    _run_command(["uv", "sync"])

@app.command()
def clean():
    """Deleta todos os arquivos Python compilados (__pycache__)."""
    console.print("[bold yellow]Limpando arquivos .pyc e __pycache__...[/bold yellow]")
    for path in ROOT_DIR.rglob("__pycache__"):
        console.print(f"Removendo {path}...")
        shutil.rmtree(path)
    for path in ROOT_DIR.rglob("*.py[co]"):
        console.print(f"Removendo {path}...")
        path.unlink()
    console.print("[bold green]Limpeza concluída![/bold green]")

@app.command()
def clean_artifacts():
    """Limpa artefatos do projeto (datasets, modelos, logs)."""
    console.print("[bold yellow]Limpando artefatos (dados, modelos, logs)...[/bold yellow]")
    dirs_to_clean = [
        ROOT_DIR / "data" / "bronze",
        ROOT_DIR / "data" / "silver",
        ROOT_DIR / "data" / "gold",
        ROOT_DIR / "models",
        ROOT_DIR / "logs",
    ]
    for d in dirs_to_clean:
        if d.exists():
            console.print(f"Limpando o diretório: {d}")
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True) # Recria o diretório vazio
    console.print("[bold green]Limpeza de artefatos concluída![/bold green]")

@app.command()
def lint():
    """Verifica a formatação e o linting do código com ruff."""
    console.print("[bold blue]Verificando formatação com ruff...[/bold blue]")
    _run_command(["uv", "run", "ruff", "format", "--check"])
    console.print("[bold blue]Verificando linting com ruff...[/bold blue]")
    _run_command(["uv", "run", "ruff", "check"])
    console.print("[bold green]Verificações de linting concluídas![/bold green]")

@app.command()
def format():
    """Formata o código com ruff."""
    console.print("[bold blue]Formatando código com ruff...[/bold blue]")
    _run_command(["uv", "run", "ruff", "check", "--fix"])
    _run_command(["uv", "run", "ruff", "format"])
    console.print("[bold green]Formatação concluída![/bold green]")

@app.command()
def test():
    """Executa os testes com pytest."""
    console.print("[bold magenta]Executando testes...[/bold magenta]")
    _run_command(["uv", "run", "pytest", "tests/"])

@app.command()
def create_environment():
    """Cria um ambiente virtual Python com uv."""
    console.print(f"[bold cyan]Criando ambiente virtual com Python {PYTHON_VERSION}...[/bold cyan]")
    _run_command(["uv", "venv", "--python", PYTHON_VERSION])
    console.print("[bold green]Ambiente virtual criado com sucesso![/bold green]")
    console.print("Para ativar, use:")
    console.print("[cyan]>>> Windows: .\\.venv\\Scripts\\activate[/cyan]")
    console.print("[cyan]>>> Unix/macOS: source ./.venv/bin/activate[/cyan]")

# --- Comandos do Pipeline ---

@app.command()
def bronze():
    """Executa a ingestão de dados para a camada Bronze."""
    console.print("[bold]Executando ingestão para a camada Bronze...[/bold]")
    _run_command(["uv", "run", "python", "data_processing/bronze/ingest_bronze.py"])

@app.command()
def silver():
    """Executa a limpeza de dados para a camada Silver."""
    console.print("[bold]Executando limpeza para a camada Silver...[/bold]")
    _run_command(["uv", "run", "python", "data_processing/silver/clean_data.py"])

@app.command()
def validate():
    """Executa a validação de dados da camada Silver."""
    console.print("[bold]Executando validação da camada Silver...[/bold]")
    _run_command(["uv", "run", "python", "data_processing/silver/validate_data.py"])

@app.command()
def gold():
    """Cria features na camada Gold."""
    console.print("[bold]Criando features na camada Gold...[/bold]")
    validate() # Garante que a validação seja executada antes
    _run_command(["uv", "run", "python", "data_processing/gold/build_features.py"])
    
@app.command(name="pipeline")
def run_pipeline():
    """Executa o pipeline de dados completo: Bronze -> Silver -> Validate -> Gold."""
    console.print("[bold purple]>>> Iniciando pipeline de dados completo...[/bold purple]")
    requirements()
    bronze()
    silver()
    gold()
    train()
    console.print("[bold purple]>>> Pipeline completo executado com sucesso![/bold purple]")

# --- Comandos do Projeto ---

@app.command()
def train():
    """Treina o modelo de machine learning."""
    console.print("[bold yellow]Iniciando treinamento do modelo...[/bold yellow]")
    requirements()
    _run_command(["uv", "run", "python", "ml_classification/modeling/train.py"])


if __name__ == "__main__":
    app()