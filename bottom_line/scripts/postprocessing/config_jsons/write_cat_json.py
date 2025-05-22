import argparse
import json

def generate_categories(recoVar, binName, boundaries, isData=False):
    lead_mvaId_string = "lead_mvaID"
    sublead_mvaId_string = "sublead_mvaID"
    sigma_m_over_m_string = "sigma_m_over_m_corr_smeared_decorr"
    if isData: 
        sigma_m_over_m_string = "sigma_m_over_m_smeared_decorr"
        
    categories = {}
    mass_resolution_categories = ['cat0', 'cat1', 'cat2']
    mass_resolution_thresholds = [0.010, 0.014]
    
    absolute_value_vars = ["rapidity", "first_jet_y"]
    
    for i in range(len(boundaries) - 1):
        for j, mass_resolution_cat in enumerate(mass_resolution_categories):
            if binName == "":
                binName_str = recoVar
            else:
                binName_str = binName
            category_name = f"RECO_{binName_str}_{str(boundaries[i]).replace('.', 'p')}_{str(boundaries[i + 1]).replace('.', 'p')}_{mass_resolution_cat}"
            if recoVar in absolute_value_vars:
                # Apply parquet way of handling logic AND and OR
                # Outer list is always OR while inner lists are AND conditions
                if boundaries[i] != 0:
                    category_filters = [
                        [
                            [lead_mvaId_string, ">", 0.25],
                            [sublead_mvaId_string, ">", 0.25],
                            [recoVar, ">=", boundaries[i]],
                            [recoVar, "<", boundaries[i + 1]]
                    ],
                    [
                        [lead_mvaId_string, ">", 0.25],
                        [sublead_mvaId_string, ">", 0.25],
                        [recoVar, "<=", -1 * boundaries[i]],
                        [recoVar, ">", -1 * boundaries[i + 1]]
                    ]
                ]
                
                else:
                    category_filters = [
                        [
                            [lead_mvaId_string, ">", 0.25],
                            [sublead_mvaId_string, ">", 0.25], 
                            [recoVar, ">=", boundaries[i]],
                            [recoVar, "<", boundaries[i + 1]]
                        ],
                        [
                            [lead_mvaId_string, ">", 0.25],
                            [sublead_mvaId_string, ">", 0.25], 
                            [recoVar, "<", boundaries[i]],
                            [recoVar, ">=", -1 * boundaries[i + 1]]
                        ]
                ]
            
            
            
                for k, _ in enumerate(category_filters):
                    if j == 0:
                        category_filters[k].append([sigma_m_over_m_string, "<", mass_resolution_thresholds[j]])
                    elif j == len(mass_resolution_categories) - 1:
                        category_filters[k].append([sigma_m_over_m_string, ">=", mass_resolution_thresholds[j - 1]])
                    else:
                        category_filters[k].append([sigma_m_over_m_string, ">=", mass_resolution_thresholds[j - 1]])
                        category_filters[k].append([sigma_m_over_m_string, "<", mass_resolution_thresholds[j]])
                    
            else:
                category_filters = [
                    [lead_mvaId_string, ">", 0.25],
                    [sublead_mvaId_string, ">", 0.25],
                    [recoVar, ">=", boundaries[i]],
                    [recoVar, "<", boundaries[i + 1]]
                ]
                if j == 0:
                    category_filters.append([sigma_m_over_m_string, "<", mass_resolution_thresholds[j]])
                elif j == len(mass_resolution_categories) - 1:
                    category_filters.append([sigma_m_over_m_string, ">=", mass_resolution_thresholds[j - 1]])
                else:
                    category_filters.append([sigma_m_over_m_string, ">=", mass_resolution_thresholds[j - 1]])
                    category_filters.append([sigma_m_over_m_string, "<", mass_resolution_thresholds[j]])
            
            categories[category_name] = {
                "cat_filter": category_filters
            }
    return categories

def save_to_json(output_file, categories):
    with open(output_file, 'w') as outfile:
        json.dump(categories, outfile, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Generate JSON file with specified reco variable boundaries.')
    parser.add_argument('output_file', help='Output file name and location')
    parser.add_argument('recoVar', type=str, default="pt", help="Reco variable")
    parser.add_argument('boundaries', nargs='+', type=float, help='Boundaries list')
    parser.add_argument('--isData', action="store_true", default=False, help="Add this flag if you are running over data, this changes the name of the sigma_m/m variable that is read.")
    parser.add_argument("--binName", dest="binName", type=str, required=False, default="")
    args = parser.parse_args()

    print(args.isData)

    categories = generate_categories(args.recoVar, args.binName, args.boundaries, args.isData)
    save_to_json(args.output_file, categories)

if __name__ == "__main__":
    main()
