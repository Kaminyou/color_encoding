import sys
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import argparse
from .utils import find_read_based_on_scaffold,make_read_scaffold_df,
from .utils import make_read_selected_matrix,image_pileup,PIXEL_MAX,NO_MATCH_REF,FINAL_RGB

def consider_read_of_one_scaffold_by_bed_region(fasta_df, sam, bed_name, scaffold_name,
                                                core, window, depth, expand, output, 
                                                all_torch_included = 0, only_find_the_longest=True):
    # Make a 3 dimension image
    BIAS_OF_READ = 0 #Customerization
    MAPPING_QUALITY_NOT_PROVIDED = 255 #Customerization
    ON_POSITIVE_STRAND_NOT_PROVIDED = 255 #Customerization
    bed = read_bed(bed_name)
    bed_df = pd.DataFrame(bed)
    bed_df = bed_df[[0,1,2,3]]
    bed_df.columns = ["scaffold","start","end","label"]
    
    if scaffold_name != "all_scaffold":
        
        bed_selected_df = bed_df[bed_df["scaffold"]==scaffold_name]
        if len(bed_selected_df) == 0:
            print("There is no scaffold named: ",scaffold_name)
            print("Please choose the scaffold below:")
            print(bed_df["scaffold"].unique())
            sys.exit("ERROR! known scaffold name")
        
        total_process = len(bed_selected_df)
        right_process = 1

        start_time = datetime.now()
        scaffold_image_dict = {}
        scaffold_length = fasta_df[fasta_df["scaffold"] == scaffold_name]["len"].values[0]

        # Select useful information in dafaframe of selected scaffold
        selected_scaffold_df = find_read_based_on_scaffold(scaffold_name, sam)

        if len(selected_scaffold_df) == 0:
            print("no covered reads")
            print("==========================================")
            sys.exit("ERROR! no covered reads")
            

        to_use_selected_scaffold_df = make_read_scaffold_df(fasta_df, selected_scaffold_df, scaffold_name, expand)
        if len(to_use_selected_scaffold_df) == 0:
            print("no covered reads")
            print("==========================================")
            sys.exit("ERROR! no covered reads")
        to_use_selected_scaffold_df = to_use_selected_scaffold_df.sort_values(by=["start"]) # Sort the reads by start locus


        for bed_index in bed_selected_df.index:
            start = int(bed_df.iloc[bed_index]["start"])
            end = int(bed_df.iloc[bed_index]["end"])
            label = bed_df.iloc[bed_index]["label"]
            ######
            fragment = np.arange((start+1),(end+1))

            searching_interval = fragment
            skip_range = fragment[0] #Customerization # MUST START FROM 1
            start_time_fragment = datetime.now()

            store_name = scaffold_name + "|" + str(fragment[0]) + " to " + str(fragment[-1]) + "|"+label
            print("Start process ",right_process," / ",total_process," : ",store_name)
            print("Step 1: finding reads", end='')
            start_time_temp = datetime.now()
            read_selected_matrix = make_read_selected_matrix(scaffold_name, 
                                                             searching_interval, 
                                                             to_use_selected_scaffold_df, 
                                                             all_torch_included,
                                                             window, skip_range, 
                                                             depth, only_find_the_longest, core)
            end_time_temp  = datetime.now()
            print(" | Duration: {}".format(end_time_temp - start_time_temp))

            print("Step 2: image pileup", end='')
            start_time_temp = datetime.now()
            pre_image_base_color, pre_image_mapping_quality = image_pileup(scaffold_name,fasta_df, 
                                                                           read_selected_matrix, 
                                                                           to_use_selected_scaffold_df, 
                                                                           window, depth,
                                                                           skip_range,searching_interval, 
                                                                           PIXEL_MAX, BIAS_OF_READ, 
                                                                           MAPPING_QUALITY_NOT_PROVIDED, 
                                                                           ON_POSITIVE_STRAND_NOT_PROVIDED, 
                                                                           NO_MATCH_REF, core)
            end_time_temp  = datetime.now()
            print(" | Duration: {}".format(end_time_temp - start_time_temp))

            print("Step 3: combination", end='')
            start_time_temp = datetime.now()
            final_rgb = channels_to_RGB(pre_image_base_color, pre_image_mapping_quality)
            end_time_temp  = datetime.now()
            print(" | Duration: {}".format(end_time_temp - start_time_temp))

            # Save to dictionary
            print("Step 4: save to dictionary", end='')


            start_time_temp = datetime.now()
            scaffold_image_dict[store_name] = final_rgb
            end_time_temp  = datetime.now()
            print(" | Duration: {}".format(end_time_temp - start_time_temp))


            end_time_fragment = datetime.now()
            right_process += 1

            print("finish ",store_name, end='')
            print(" | Duration: {}".format(end_time_fragment - start_time_fragment))
            print("==========================================")


            print("==================================================================================")

        print("Final stage: save to pickle")
        # Store data to pickle(serialize)

        with open(output, "wb") as handle:
            pickle.dump(scaffold_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_time = datetime.now()
        print("Total duration: {}".format(end_time - start_time))
    
    else:
        total_process = len(bed_df)
        right_process = 1
        for scaffold_name in bed_df["scaffold"].unique():
            bed_selected_df = bed_df[bed_df["scaffold"]==scaffold_name]

            start_time = datetime.now()
            scaffold_image_dict = {}
            scaffold_length = fasta_df[fasta_df["scaffold"] == scaffold_name]["len"].values[0]

            # Select useful information in dafaframe of selected scaffold
            selected_scaffold_df = find_read_based_on_scaffold(scaffold_name, sam)

            if len(selected_scaffold_df) == 0:
                print("no covered reads")
                print("==========================================")
                continue
            to_use_selected_scaffold_df = make_read_scaffold_df(fasta_df, selected_scaffold_df, scaffold_name, expand)
            if len(to_use_selected_scaffold_df) == 0:
                print("no covered reads")
                print("==========================================")
                continue
            to_use_selected_scaffold_df = to_use_selected_scaffold_df.sort_values(by=["start"]) # Sort the reads by start locus


            for bed_index in bed_selected_df.index:
                start = int(bed_df.iloc[bed_index]["start"])
                end = int(bed_df.iloc[bed_index]["end"])
                label = bed_df.iloc[bed_index]["label"]
                ######
                fragment = np.arange(start,(end+1))

                searching_interval = fragment
                skip_range = fragment[0] #Customerization # MUST START FROM 1
                start_time_fragment = datetime.now()

                store_name = scaffold_name + "|" + str(fragment[0]) + " to " + str(fragment[-1]) + "|"+label
                print("Start process ",right_process," / ",total_process," : ",store_name)
                print("Step 1: finding reads", end='')
                start_time_temp = datetime.now()
                read_selected_matrix = make_read_selected_matrix(scaffold_name, 
                                                                 searching_interval, 
                                                                 to_use_selected_scaffold_df, 
                                                                 all_torch_included,
                                                                 window, skip_range, 
                                                                 depth, only_find_the_longest, core)
                end_time_temp  = datetime.now()
                print(" | Duration: {}".format(end_time_temp - start_time_temp))

                print("Step 2: image pileup", end='')
                start_time_temp = datetime.now()
                pre_image_base_color, pre_image_mapping_quality = image_pileup(scaffold_name,
                                         fasta_df, 
                                         read_selected_matrix, 
                                         to_use_selected_scaffold_df, 
                                         window, 
                                         depth,
                                         skip_range,
                                         searching_interval, 
                                         PIXEL_MAX, 
                                         BIAS_OF_READ, 
                                         MAPPING_QUALITY_NOT_PROVIDED, 
                                         ON_POSITIVE_STRAND_NOT_PROVIDED, 
                                         NO_MATCH_REF, core)
                end_time_temp  = datetime.now()
                print(" | Duration: {}".format(end_time_temp - start_time_temp))

                print("Step 3: combination", end='')
                start_time_temp = datetime.now()
                FINAL_RGB = channels_to_RGB(PIXEL_MAX_empty_def,
                                        pre_image_base_color, 
                                        pre_image_mapping_quality)
                end_time_temp  = datetime.now()
                print(" | Duration: {}".format(end_time_temp - start_time_temp))

                # Save to dictionary
                print("Step 4: save to dictionary", end='')


                start_time_temp = datetime.now()
                scaffold_image_dict[store_name] = FINAL_RGB
                end_time_temp  = datetime.now()
                print(" | Duration: {}".format(end_time_temp - start_time_temp))


                end_time_fragment = datetime.now()
                right_process += 1

                print("finish ",store_name, end='')
                print(" | Duration: {}".format(end_time_fragment - start_time_fragment))
                print("==========================================")


            print("==================================================================================")

        print("Final stage: save to pickle")
        # Store data to pickle(serialize)

        with open(output, "wb") as handle:
            pickle.dump(scaffold_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_time = datetime.now()
        print("Total duration: {}".format(end_time - start_time))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--input_fasta", help="the input fasta file",required=True)
    parser.add_argument("-is", "--input_sam",  help="the input sam file",required=True)
    parser.add_argument("-ib", "--input_bed",  help="the input bed file",required=True)
    parser.add_argument("-o", "--output", help="the output image file name, must end with .pickle",required=True)
    parser.add_argument("-w", "--window",type=int, default=65, help="the window of the output image")
    parser.add_argument("-s", "--scaffold", default = "all_scaffold", help="one scaffold you want to make image")
    parser.add_argument("-c", "--core", type=int, default=40, help="cores you want to use")
    parser.add_argument("-d", "--depth", type=int, default=2, help="depth you want to use")
    parser.add_argument("-e", "--expand", type=int, default=0, help="expand you want to use")

    args = parser.parse_args()

    fasta_df = read_fasta(args.input_fasta)
    sam = read_sam(args.input_sam)

    consider_read_of_one_scaffold_by_bed_region(fasta_df, sam, args.input_bed, args.scaffold,
                                                args.core, args.window, args.depth, args.expand, args.output)
