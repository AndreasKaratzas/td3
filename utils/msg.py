
import sys
sys.path.append('../')

from utils.functions import colorstr


def print_training_message(
        agent: str,
        env_id: str,
        epochs: int,
        device: str,
        elite_metric: str,
        auto_save: bool,
        parent_dir_printable_version: str,
        project_path_printable_version: str
    ):
        new_line = '\n'
        tab_char = '\t'

        print(f"\n\n\t\tTraining a {colorstr(options=['red', 'underline'], string_args=list([agent]))}\n"
              f"\t\t      in {colorstr(options=['red', 'underline'], string_args=list([env_id]))} environment for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} epochs using\n"
              f"\t\t    a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}. {colorstr(options=['blue', 'bold'], string_args=list(['Odysseus']))} will select checkpoints \n"
              f"\t\t        based on {colorstr(options=['red', 'underline'], string_args=list([elite_metric]))}. Auto-saving is {colorstr(options=['blue'], string_args=list(['enabled'])) if auto_save else colorstr(options=['blue'], string_args=list(['disabled']))}{' and' + new_line + tab_char + tab_char + '         the agent will begin learning from step ' + colorstr(options=['red', 'underline'], string_args=list([''])) + '.' + new_line if False else '.' + new_line}"
              f"\n\n\t               The experiment logger is uploaded locally at: \n"
              f"  {colorstr(options=['blue', 'underline'], string_args=list([parent_dir_printable_version]))}."
              f"\n\n\t\t                Project absolute path is:\n\t"
              f"{colorstr(options=['blue', 'underline'], string_args=list([project_path_printable_version]))}.\n\n")

def print_test_message(
        agent: str, 
        env_id: str, 
        epochs: int, 
        device: str,
        parent_dir_printable_version: str, 
        project_path_printable_version: str
    ):

    print(f"\n\n\t\t    Evaluating a {colorstr(options=['red', 'underline'], string_args=list([agent]))}\n"
            f"\t\t      in {colorstr(options=['red', 'underline'], string_args=list([env_id]))} environment for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} episodes using \n"
            f"\t\t                 a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}."
            f"\n\n\t               The experiment logger is uploaded locally at: \n"
            f"  {colorstr(options=['blue', 'underline'], string_args=list([parent_dir_printable_version]))}."
            f"\n\n\t\t                Project absolute path is:\n\t"
            f"{colorstr(options=['blue', 'underline'], string_args=list([project_path_printable_version]))}.\n\n")


def info():
    print(f"\n\n"
          f"\t\t        The {colorstr(['red', 'bold'], list(['Odysseus']))} suite serves as a framework for \n"
          f"\t\t    reinforcement learning agents training on OpenAI GYM \n"
          f"\t\t     defined environments. This repository was created \n"
          f"\t\t    created to help in future projects that would require \n"
          f"\t\t     such agents to find solutions for complex problems. \n"
          f"\n")
