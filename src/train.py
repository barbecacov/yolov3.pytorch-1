import argparse


def parse_arg():
  parser = argparse.ArgumentParser(description='YOLO v3 training')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_arg()
  
