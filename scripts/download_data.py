"""
Data Download Script for PASTO
Downloads and prepares educational datasets
"""

import argparse
import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def download_file(url: str, destination: Path):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_oulad(data_dir: Path):
    """
    Download Open University Learning Analytics Dataset
    
    Args:
        data_dir: Directory to save data
    """
    print("\n" + "="*70)
    print("DOWNLOADING OULAD DATASET")
    print("="*70)
    
    oulad_dir = data_dir / 'oulad'
    oulad_dir.mkdir(parents=True, exist_ok=True)
    
    # OULAD is available from Kaggle
    print("\nNote: OULAD dataset requires manual download from:")
    print("https://analyse.kmi.open.ac.uk/open_dataset")
    print("\nAlternatively, use Kaggle:")
    print("https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad")
    
    # Check if already downloaded
    required_files = [
        'studentInfo.csv',
        'studentAssessment.csv',
        'studentVle.csv',
        'courses.csv',
        'assessments.csv',
        'vle.csv'
    ]
    
    all_exist = all((oulad_dir / f).exists() for f in required_files)
    
    if all_exist:
        print("\n✓ OULAD dataset already exists")
        return
    
    # Provide instructions for manual download
    print("\nPlease download the OULAD dataset manually and place the following files in:")
    print(f"  {oulad_dir}")
    print("\nRequired files:")
    for f in required_files:
        print(f"  - {f}")
    
    print("\nAfter downloading, run this script again.")


def create_synthetic_data(data_dir: Path, num_students: int = 1000):
    """
    Create synthetic student data for testing
    
    Args:
        data_dir: Directory to save data
        num_students: Number of synthetic students
    """
    print("\n" + "="*70)
    print("CREATING SYNTHETIC DATASET")
    print("="*70)
    
    synthetic_dir = data_dir / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    np.random.seed(42)
    
    # Student info
    print(f"\nGenerating {num_students} synthetic students...")
    
    student_info = pd.DataFrame({
        'id_student': range(num_students),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], num_students),
        'code_presentation': np.random.choice(['2013J', '2014B'], num_students),
        'gender': np.random.choice(['M', 'F'], num_students),
        'region': np.random.choice(['Region' + str(i) for i in range(10)], num_students),
        'highest_education': np.random.choice(['A Level', 'HE Qualification', 'Lower Than A Level'], num_students),
        'imd_band': np.random.choice(['0-10%', '10-20%', '20-30%', '30-40%', '40-50%'], num_students),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], num_students),
        'num_of_prev_attempts': np.random.randint(0, 4, num_students),
        'studied_credits': np.random.choice([60, 120, 180, 240], num_students),
        'disability': np.random.choice(['N', 'Y'], num_students, p=[0.85, 0.15]),
        'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'], num_students, p=[0.4, 0.2, 0.25, 0.15])
    })
    
    # VLE interactions
    print("Generating VLE interaction data...")
    vle_records = []
    
    for student_id in tqdm(range(num_students), desc="VLE data"):
        module = student_info.loc[student_id, 'code_module']
        presentation = student_info.loc[student_id, 'code_presentation']
        
        # Generate 20-40 weeks of data
        num_weeks = np.random.randint(20, 41)
        
        for week in range(num_weeks):
            # Number of clicks decreases over time for dropouts
            if student_info.loc[student_id, 'final_result'] in ['Withdrawn', 'Fail']:
                base_clicks = max(0, 100 - week * 3)
            else:
                base_clicks = 100
            
            clicks = max(0, int(np.random.normal(base_clicks, 30)))
            
            vle_records.append({
                'code_module': module,
                'code_presentation': presentation,
                'id_student': student_id,
                'date': week * 7,  # Convert to days
                'sum_click': clicks,
                'week': week
            })
    
    student_vle = pd.DataFrame(vle_records)
    
    # Assessment data
    print("Generating assessment data...")
    assessment_records = []
    
    for student_id in tqdm(range(num_students), desc="Assessment data"):
        module = student_info.loc[student_id, 'code_module']
        presentation = student_info.loc[student_id, 'code_presentation']
        
        # 5-10 assessments per student
        num_assessments = np.random.randint(5, 11)
        
        for assess_id in range(num_assessments):
            # Score depends on final result
            if student_info.loc[student_id, 'final_result'] == 'Distinction':
                score = np.random.normal(85, 5)
            elif student_info.loc[student_id, 'final_result'] == 'Pass':
                score = np.random.normal(65, 10)
            else:
                score = np.random.normal(45, 15)
            
            score = np.clip(score, 0, 100)
            
            assessment_records.append({
                'id_assessment': assess_id,
                'code_module': module,
                'code_presentation': presentation,
                'id_student': student_id,
                'date': assess_id * 20,  # Assessments spread out
                'score': score,
                'weight': 100 / num_assessments
            })
    
    student_assessment = pd.DataFrame(assessment_records)
    
    # Course info
    courses = pd.DataFrame({
        'code_module': ['AAA', 'BBB', 'CCC', 'DDD'],
        'code_presentation': ['2013J'] * 4,
        'module_presentation_length': [269, 241, 235, 260]
    })
    
    # Assessment metadata
    assessments = pd.DataFrame({
        'id_assessment': range(10),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], 10),
        'code_presentation': ['2013J'] * 10,
        'assessment_type': np.random.choice(['TMA', 'CMA', 'Exam'], 10),
        'date': [i * 20 for i in range(10)],
        'weight': [10] * 10
    })
    
    # VLE metadata
    vle = pd.DataFrame({
        'id_site': range(20),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], 20),
        'code_presentation': ['2013J'] * 20,
        'activity_type': np.random.choice(['resource', 'oucontent', 'url', 'forum'], 20),
        'week_from': np.random.randint(0, 30, 20),
        'week_to': np.random.randint(0, 30, 20)
    })
    
    # Save all files
    print("\nSaving synthetic dataset...")
    student_info.to_csv(synthetic_dir / 'studentInfo.csv', index=False)
    student_vle.to_csv(synthetic_dir / 'studentVle.csv', index=False)
    student_assessment.to_csv(synthetic_dir / 'studentAssessment.csv', index=False)
    courses.to_csv(synthetic_dir / 'courses.csv', index=False)
    assessments.to_csv(synthetic_dir / 'assessments.csv', index=False)
    vle.to_csv(synthetic_dir / 'vle.csv', index=False)
    
    print(f"\n✓ Synthetic dataset created in: {synthetic_dir}")
    print(f"  Students: {num_students}")
    print(f"  VLE records: {len(vle_records)}")
    print(f"  Assessment records: {len(assessment_records)}")


def verify_dataset(data_dir: Path):
    """
    Verify dataset integrity
    
    Args:
        data_dir: Dataset directory
    """
    print("\n" + "="*70)
    print("VERIFYING DATASET")
    print("="*70)
    
    required_files = [
        'studentInfo.csv',
        'studentAssessment.csv',
        'studentVle.csv',
        'courses.csv',
        'assessments.csv',
        'vle.csv'
    ]
    
    all_good = True
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"✓ {file}: {len(df)} records")
            except Exception as e:
                print(f"✗ {file}: Error reading - {e}")
                all_good = False
        else:
            print(f"✗ {file}: Not found")
            all_good = False
    
    if all_good:
        print("\n✓ All required files present and readable")
    else:
        print("\n✗ Some files are missing or unreadable")
    
    return all_good


def main(args):
    """Main download function"""
    
    data_root = Path(args.output_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'oulad':
        download_oulad(data_root)
        
        # If OULAD doesn't exist, offer to create synthetic data
        oulad_dir = data_root / 'oulad'
        if not verify_dataset(oulad_dir):
            print("\nOULAD dataset not found.")
            response = input("Create synthetic data for testing? (y/n): ")
            if response.lower() == 'y':
                create_synthetic_data(data_root, num_students=args.num_synthetic)
    
    elif args.dataset == 'synthetic':
        create_synthetic_data(data_root, num_students=args.num_synthetic)
    
    elif args.dataset == 'indian_school':
        print("\nIndian School dataset requires manual download.")
        print("Please obtain the dataset and place it in:")
        print(f"  {data_root / 'indian_school'}")
    
    else:
        print(f"Unknown dataset: {args.dataset}")
        return
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets for PASTO')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='oulad',
        choices=['oulad', 'synthetic', 'indian_school'],
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Output directory for data'
    )
    
    parser.add_argument(
        '--num_synthetic',
        type=int,
        default=1000,
        help='Number of synthetic students to generate'
    )
    
    args = parser.parse_args()
    
    main(args)
