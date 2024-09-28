# Unit test für neural networks hands on

import pytest
from pytest import approx
import torch
from torch import nn

from IPython.display import display, HTML

# Define the HTML and CSS for the green info box with a smiley
message = """
<div style="
    padding: 10px;
    border-radius: 5px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    font-size: 16px;
    font-family: Arial, sans-serif;
    margin: 10px 0;
">
    <span style="font-size: 20px;">&#128512;</span>  <!-- Smiley emoji -->
    <strong>Gut gemacht!</strong> <!-- Text -->
</div>
"""

# Display the message


# Aufgabe 1

def test_aufgabe1(loc):
    ds = loc['ds']
    assert isinstance(ds, torch.utils.data.Dataset), "DS ist nicht vom richtigen Typ"
    assert hasattr(ds,'__len__'), "__len__() Funktion ist nicht definiert."
    assert hasattr(ds,'__getitem__'), "__getitem__() Funktion ist nicht definiert."
    assert hasattr(ds,'__init__'), "__init__() Funktion ist nicht definiert."
    assert len(ds)==73257, "Datensatz hat nicht die richtige Länge"
    assert len(ds[0])==2, "Datensatz.__getitem__() gibt keine 2 Werte zurück"
    assert isinstance(ds[0][0],torch.Tensor), "Datensatz.__getitem__() gibt keinen Tensor als ersten Rückgabewert"
    assert len(ds[0][0].shape)==3, "Datensatz.__getitem__() gibt für Eingabedaten falsche Dimensionen zurück (erwartet: 3x32x32)"
    assert ds[0][0].shape[0]==3, "Datensatz.__getitem__() gibt für Eingabedaten falsche Dimensionen zurück (erwartet: 3x32x32)"
    assert ds[0][0].shape[1]==32, "Datensatz.__getitem__() gibt für Eingabedaten falsche Dimensionen zurück (erwartet: 3x32x32)"
    assert ds[0][0].shape[2]==32, "Datensatz.__getitem__() gibt für Eingabedaten falsche Dimensionen zurück (erwartet: 3x32x32)"
    display(HTML(message))


def test_aufgabe2(loc):
    sample = next(iter(loc['data_loader']))[0]

    assert sample.shape[0]==32, 'Batch size soll 32 sein'
    assert sample.shape[1]==3, 'Tensor soll Dimension Bx3x32x32 haben'
    assert sample.shape[2]==32, 'Tensor soll Dimension Bx3x32x32 haben'
    assert sample.shape[3]==32, 'Tensor soll Dimension Bx3x32x32 haben'
    display(HTML(message))


def test_aufgabe3(loc):
    model=loc['model']
    assert isinstance(model, nn.Module), "Falscher Objekttyp für MLP. Muss nn.Module sein."
    assert len(list(model.modules()))==6, "Falsche Anzahl an Layern. Insgesamt 5 Layer notwendig."

    assert list(model.modules())[1].in_features==3072, "Eingangsgröße (muss sein: 32x32x3 = 3072) stimmt nicht."
    assert list(model.modules())[1].out_features==256, "Hidden Neurons im ersten Layer ist nicht 256"
    assert list(model.modules())[2].out_features==128, "Hidden Neurons im zweiten Layer ist nicht 128"
    assert list(model.modules())[3].out_features==64, "Hidden Neurons im dritten Layer ist nicht 64"
    display(HTML(message))

