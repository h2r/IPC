import os
from shutil import copytree, ignore_patterns
import shutil
from zipfile import ZipFile
import os



ign_pats = [".git*",".gitignore", "scoping.egg-info/*", ".vscode", ".ds_store", "ignore*", "ubuntu_scripts_new_planutils/*"]
repo_root = os.path.dirname(__file__)
repo_parent = os.path.dirname(repo_root)
tgt_dir = os.path.join(repo_parent, "scoping_supplement")
tgt_dir_repo = os.path.join(tgt_dir, "scoping")

if os.path.isdir(tgt_dir):
    shutil.rmtree(tgt_dir)



copytree(repo_root, tgt_dir_repo, ignore=ignore_patterns(*ign_pats))

supplement_pdf_pth = os.path.join(repo_parent, "Scoping_AAAI_2023_Supplement.pdf")
shutil.copy(supplement_pdf_pth, tgt_dir)

archive_pth = os.path.join(repo_parent, "scoping_supplement")
archive_pth_with_ext = os.path.join(archive_pth, "zip")
if os.path.isfile(archive_pth_with_ext):
    os.remove(archive_pth_with_ext)
shutil.make_archive(archive_pth, format="zip", root_dir=repo_parent, base_dir="scoping_supplement")