
from pathlib import Path
import pandas as pd
import glob
import pandas_datareader.data as web
import re
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import locale
locale.setlocale( locale.LC_ALL, '' )


año1=mn.año1
closes_activa=mn.closesa
