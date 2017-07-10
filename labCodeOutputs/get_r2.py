#!/usr/bin/env python3

"""
Prints label (alphabetically sorted), r2 score.
"""

import os

def main():
  dirs = sorted([x for x in os.listdir('.') if os.path.isdir(x)])
  for d in dirs:
    with open(os.path.join(d, 'stdout.txt'), 'r') as f:
      lines = f.readlines()
    r2_line = lines[-2].strip()
    if r2_line[:5] != 'R2 = ':
      raise Exception('Unexpected r2_line:', r2_line)
    print(d + ',' + r2_line[5:])


if __name__ == '__main__':
    main()

