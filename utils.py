import numpy as np
import matplotlib.pyplot as pt
import csv
import pandas as pd
import cantools 
import matplotlib.animation as animation
from matplotlib import style
from haversine import haversine, Unit
import time
import strym as s
import sys

from bagpy import bagreader
db2 = cantools.database.Database()

import os
from scipy import interpolate
from copy import deepcopy


def get_bagfile_timeseries(bag_file_name):
	# Should change here if you want different topics that what this function grabs:
	file_names = os.listdir()
	file_processed = False
	if(bag_file_name[:-4] in file_names):file_processed = True
	
	if(file_processed):
		print('File already processed.')
		processed_path = os.path.join(os.getcwd(),bag_file_name[:-4])
		
		accel_data = pd.read_csv(os.path.join(processed_path,'accel.csv'))
		cmd_accel_data = pd.read_csv(os.path.join(processed_path,'cmd_accel.csv'))
		controls_allowed_data = pd.read_csv(os.path.join(processed_path,'car-libpanda-controls_allowed.csv'))
		lead_dist_data = pd.read_csv(os.path.join(processed_path,'lead_dist.csv'))
		rv_data = pd.read_csv(os.path.join(processed_path,'rel_vel.csv'))
		vel_data = pd.read_csv(os.path.join(processed_path,'vel.csv'))
		
	else:
		print('Loading bag file: '+bag_file_name)
		b = bagreader(bag_file_name)
		print('Extracting relevant quantities')
		accel_csv = b.message_by_topic(topic='/accel')
		accel_data = pd.read_csv(accel_csv)

		cmd_accel_csv = b.message_by_topic(topic='/cmd_accel')
		cmd_accel_data = pd.read_csv(cmd_accel_csv)

		controls_allowed_csv = b.message_by_topic(topic='/car/libpanda/controls_allowed')
		controls_allowed_data = pd.read_csv(controls_allowed_csv)

		cbf_debug_csv = b.message_by_topic(topic='/cbf/cbf_debug')
		cbf_debug_data = pd.read_csv(cbf_debug_csv)

		accel_pre_csv = b.message_by_topic(topic='/cmd_accel_pre')
		accel_pre_data = pd.read_csv(accel_pre_csv)

		accel_safe_csv = b.message_by_topic(topic='/cmd_accel_safe')
		accel_safe_data = pd.read_csv(accel_safe_csv)

		lead_dist_csv = b.message_by_topic(topic='/lead_dist')
		lead_dist_data = pd.read_csv(lead_dist_csv)

		rv_csv = b.message_by_topic(topic='/rel_vel')
		rv_data = pd.read_csv(rv_csv)

		vel_csv = b.message_by_topic(topic='/vel')
		vel_data = pd.read_csv(vel_csv)

		print('Relevant data extracted: '+bag_file_name)
	
	return [accel_data,cmd_accel_data,controls_allowed_data,lead_dist_data,vel_data,rv_data]

def get_control_active_sections(controls_allowed_data):
	
	control_active_sections = [] #[begin_time,end_time,begin_index,end_index]
	control_active = False

	begin_control_active_time = 0
	begin_control_actve_index = 0
	end_control_active_time = 0
	end_control_active_index = 0


	for i in range(1,len(controls_allowed_data)):
		curr_control_active = controls_allowed_data['data'][i]

		# begin tracking when a switch from false to true happens
		if(not control_active and curr_control_active):
			begin_control_active_time = controls_allowed_data['Time'][i]
			begin_control_active_index = i
			control_active = True

		# end tracking when switch from true to false happens:
		if(control_active and not curr_control_active):
			end_control_active_time = controls_allowed_data['Time'][i]
			end_control_active_index = i
			control_active = False
			control_active_sections.append([begin_control_active_time,
											end_control_active_time,
											begin_control_active_index,
											end_control_active_index])
			
	return control_active_sections

def resample_data_list(data_frame_list,control_active_sections,resample_fidelity=0.05):

	resampled_data_frame_list = []

	sample_times = []
	for control_active_section in control_active_sections:
		begin_time = control_active_section[0]
		end_time = control_active_section[1]

		time_samples = np.arange(begin_time,end_time,resample_fidelity)
		sample_times = np.concatenate((sample_times,time_samples),axis=0)

	for data_frame in data_frame_list:
		column_names = data_frame.columns
		data_frame_copy = pd.DataFrame(columns = column_names)
		times = data_frame['Time']
		data_frame_copy['Time'] = sample_times
		for column in column_names:
			if(column != 'Time'):
				#Perform resampling using spline:
				f_interp = interpolate.interp1d(times,data_frame[column])
				data_resampled = f_interp(sample_times)
				data_frame_copy[column] = data_resampled

		resampled_data_frame_list.append(data_frame_copy)
		
	return resampled_data_frame_list

def get_following_events(time,spacing):
	following_event_sections = [] #[begin_time,end_time,begin_index,end_index]
	prev_is_following = False
	prev_spacing = spacing[0]

	begin_following_time = time[0]
	begin_following_index = 0
	end_following_time = 0
	end_following_index = 0
	
	for i in range(1,len(time)):
		curr_is_following = spacing[i] < 200.0
		spacing_diff = spacing[i] - prev_spacing
		time_diff = time[i] - time[i-1]
		new_leader = curr_is_following and (np.abs(spacing_diff) > 2.0) or (time_diff > 1.0) #Kind of arbitrary...
		
		
		# if a new leader cuts in front between another car that was being followed:
		if(new_leader and prev_is_following):
			end_following_index = i-1
			end_following_time = time[i-1]
			following_event = [begin_following_time,
											 end_following_time,
											 begin_following_index,
											 end_following_index]
			following_event_sections.append(following_event)
			
			# Reset for a new following event:
			begin_following_time = time[i]
			begin_following_index = i	  
		# If new leader, but no previous follower then last following event is already stashed
		if(new_leader and not prev_is_following):
			begin_following_time = time[i]
			begin_following_index = i
		
		#If just switched from following to not following then stash previous following event, but don't
		#start stashing until a new leader is found.
		if(not curr_is_following and prev_is_following):
			end_following_index = i-1
			end_following_time = time[i-1]
			following_event = [begin_following_time,
											 end_following_time,
											 begin_following_index,
											 end_following_index]
			following_event_sections.append(following_event)
			

			


		#Update the previous measurements to be the current:
		prev_spacing = spacing[i]
		prev_is_following = curr_is_following
			
	return following_event_sections   

def get_following_events_dict(bag_file_name):
	data_frame_list = get_bagfile_timeseries(bag_file_name)
	controls_allowed_data = data_frame_list[2]

	control_active_sections = get_control_active_sections(controls_allowed_data)

	control_active_sections_long = []

	for control_section in control_active_sections:
		if(control_section[1]-control_section[0]>20):
			begin = control_section[2]
			end = control_section[3]
			control_active_sections_long.append(control_section)
			
	control_active_sections = control_active_sections_long

	print('control sections longer than 20 seconds: '+str(len(control_active_sections)))
	
	resampled_data_frame_list = resample_data_list(data_frame_list,control_active_sections)
	print('Resampling finished.')

	spacing = np.array(resampled_data_frame_list[3]['data'])
	relative_speed = np.array(resampled_data_frame_list[5]['linear.z'])
	speed = np.array(resampled_data_frame_list[4]['linear.x'])
	command_accel = np.array(resampled_data_frame_list[1]['data'])
	real_accel = np.array(resampled_data_frame_list[0]['data'])
	time = np.array(resampled_data_frame_list[0]['Time'])

	following_event_sections = get_following_events(time,spacing)
	following_event_sections_long = []
	for following_event in following_event_sections:
		if (following_event[1] - following_event[0]) > 5.0:
			following_event_sections_long.append(following_event)

	print('Number following events longer than 5 seconds: '+str(len(following_event_sections_long)))
	
	following_events_spacing = []
	following_events_speed = []
	following_events_relative_speed = []
	following_events_command_accel = []
	following_events_real_accel = []
	following_events_time = []

	for following_event in following_event_sections_long:
		begin_index = following_event[2]
		end_index = following_event[3]
		following_events_spacing.append(spacing[begin_index:end_index])
		following_events_speed.append(speed[begin_index:end_index])
		following_events_relative_speed.append(relative_speed[begin_index:end_index])
		following_events_command_accel.append(command_accel[begin_index:end_index])
		following_events_real_accel.append(real_accel[begin_index:end_index])
		following_events_time.append(time[begin_index:end_index])

	all_following_events_dict = dict.fromkeys(['spacing','speed','relative_speed','command_accel','real_accel','time'])
	all_following_events_dict['spacing'] = following_events_spacing
	all_following_events_dict['speed'] = following_events_speed
	all_following_events_dict['relative_speed'] = following_events_relative_speed
	all_following_events_dict['command_accel'] = following_events_command_accel
	all_following_events_dict['real_accel'] = following_events_real_accel
	all_following_events_dict['time'] = following_events_time
	
	return all_following_events_dict

def extract_all_bags_and_write_to_csv():
	bag_files = []
	all_files = os.listdir()

	for file in all_files:
		if(file[-3:] == 'bag'):
			bag_files.append(file)
	print('Number of bag files in directory: '+str(len(bag_files)))


	multi_drive_data_dict = {}

	num_bag_files = len(bag_files)
	print('Bag files to process: '+str(num_bag_files))

	num_bag_files_processed = 0

	for bag_file_name in bag_files:
		try:
			print('Processing: '+str(bag_file_name))
			all_following_events_dict = get_following_events_dict(bag_file_name)
			num_bag_files_processed += 1
			print('Bag files processed: '+str(num_bag_files_processed)+'/'+str(num_bag_files))
			multi_drive_data_dict[bag_file_name] = all_following_events_dict
			
			
		except:
			print('Problem loading bag file: '+bag_file_name)
			num_bag_files_processed += 1
			print('Bag files processed: '+str(num_bag_files_processed)+'/'+str(num_bag_files))
			
		print('Finished processing all bag files')


	bag_files_processed = list(multi_drive_data_dict.keys())

	csv_files_written = 0

	#Make the repository where data will be stored:
	data_directory = os.path.join(os.getcwd(),'following_events_data')
	os.mkdir(data_directory)


	#Extract following events from all the bag files that were processed:
	for file in bag_files_processed:
		following_events_dict = multi_drive_data_dict[file]
		
		data_keys = list(following_events_dict.keys())

		num_following_events = len(following_events_dict['time'])

		for following_event_num in range(num_following_events):
			temp_data = []
			
			for key in data_keys:
				data_list = following_events_dict[key]
				temp_data.append(np.array(data_list[following_event_num]))

			temp_data_np_arr = np.array(temp_data)

			temp_data_np_arr = temp_data_np_arr.T
			
			header_str = data_keys[0]

			for i in range(1,len(data_keys)):
				header_str = header_str + ','+data_keys[i]
				
			file_name = file_name = file[:-4] + '_'+str(following_event_num+1)+'.csv'
			
			file_name = 'following_events_data/' + file_name
			
			np.savetxt(file_name, temp_data_np_arr, delimiter=' ', header=header_str)
			
			csv_files_written += 1
			
			sys.stdout.write('\r'+'CSV files written: '+str(csv_files_written))
			 
	print()
	print('Finished writing files.')