"""The Metashape AI Classifier app: one local NiceGUI window for options, progress and result.

The app never loads a model and never imports torch. It talks to the Metashape launcher and the
worker only through the files of a session.
"""
