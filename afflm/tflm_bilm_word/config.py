"""Config: PyTorch YouCook2 RNN/LSTM Language Model
"""
import logging
import os
import json
from logging.handlers import RotatingFileHandler


__author__ = 'chaonan99'
__copyright__ = "Copyright 2018, Haonan Chen"


class Config(object):
	"""Create experiment directory"""
	def __init__(self, args=None):
		exp_par_dir, exp_name = os.path.split(self.exp_dir)
		if args is not None:
			self.run_name = args.run_name

	def initialize_exp(self):
		## Check directory layout
		exp_par_dir, exp_name = os.path.split(self.exp_dir)
		proj_dir, src_name = os.path.split(exp_par_dir)
		# assert src_name == 'src', "Layout sould be src/exp_name"
		dump_dir = os.path.join(proj_dir, 'dump')
		assert os.path.exists(dump_dir), \
			   'There should be a dump dir under project dir'
		dump_exp_dir = os.path.join(dump_dir, exp_name)
		if not os.path.exists(dump_exp_dir):
			os.makedirs(dump_exp_dir)
		rel_dump_exp_dir = os.path.relpath(dump_exp_dir)
		if os.path.exists('./dump'):
			os.remove('./dump')
		os.symlink(rel_dump_exp_dir, './dump')

	@property
	def exp_dir(self):
		return os.path.dirname(os.path.abspath(__file__))

	@property
	def save_path(self):
		return os.path.join('dump', self.run_name) \
			   if 'dump' not in self.run_name else self.run_name

	@property
	def model_save_path(self):
		return os.path.join(self.save_path, 'model.pt')

	def get_logger(self):
		log_file_dir = self.save_path
		log_file_name = os.path.join(log_file_dir, 'train.log')
		if not os.path.exists(log_file_dir):
			os.makedirs(log_file_dir)
		logger = logging.getLogger()
		# Debug = write everything
		logger.setLevel(logging.DEBUG)

		formatter = logging.Formatter('%(asctime)s :: '
		                              '%(levelname)s :: %(message)s')
		file_handler = RotatingFileHandler(log_file_name, 'a', 1000000, 1)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		steam_handler = logging.StreamHandler()
		steam_handler.setLevel(logging.INFO)
		logger.addHandler(steam_handler)

		return logger

	def get_options(self):
		with open(os.path.join(self.save_path, 'options.json')) as f:
			options = json.load(f)
		return options


def main():
	config = Config()
	config.initialize_exp()

if __name__ == '__main__':
	main()